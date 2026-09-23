
precision highp float;
precision highp int;
precision highp usampler2DArray;

#include <splatDefines>

out vec4 vRgba;
out vec2 vSplatUv;
out vec3 vNdc;
flat out uint vSplatIndex;
flat out float adjustedStdDev;

// uniform uint numSplats;
uniform vec2 renderSize;
uniform vec4 renderToViewQuat;
uniform vec3 renderToViewPos;
uniform mat3 renderToViewBasis;
uniform float maxStdDev;
uniform float minPixelRadius;
uniform float maxPixelRadius;
uniform bool enableExtSplats;
uniform bool enableCovSplats;
uniform float time;
uniform float deltaTime;
uniform bool debugFlag;
uniform float minAlpha;
uniform bool enable2DGS;
uniform bool lodInflate;
uniform float blurAmount;
uniform float preBlurAmount;
uniform float focalDistance;
uniform float apertureAngle;
// Lens model: 0 pinhole through the EWA Jacobian, 1 equidistant fisheye, 2 Brown-Conrady with
// lensParams = (k1, k2, p1, p2). Under a lens the splat projects by the unscented transform:
// seven sigma points through the lens map, their mean and covariance in pixels.
uniform int lensModel;
uniform vec4 lensParams;
uniform float clipXY;
uniform float focalAdjustment;

uniform usampler2D ordering;
uniform usampler2DArray extSplats;
uniform usampler2DArray extSplats2;

// Required by logdepthbuf_pars_vertex (normally defined in three.js #include <common>)
bool isPerspectiveMatrix( mat4 m ) {
    return m[ 2 ][ 3 ] == -1.0;
}

#include <logdepthbuf_pars_vertex>

// Whether the lens map is valid at a view-space point: in front of the camera plane, and for the
// Brown-Conrady map on the monotone part of the radial polynomial. Past the turning point the
// map folds back into the frame, which draws ghosts.
bool lensValid(vec3 v) {
    if (v.z > -1e-4) {
        return false;
    }
    if (lensModel == 2) {
        vec2 n = v.xy / -v.z;
        float r2 = dot(n, n);
        // d(r (1 + k1 r^2 + k2 r^4)) / dr = 1 + 3 k1 r^2 + 5 k2 r^4
        return 1.0 + 3.0 * lensParams.x * r2 + 5.0 * lensParams.y * r2 * r2 > 0.0;
    }
    return true;
}

// A view-space point to pixel coordinates centred on the principal point, through the lens.
vec2 lensProject(vec3 v, vec2 focal) {
    if (lensModel == 1) {
        // Equidistant fisheye: r = f * theta, theta the angle from the optical axis.
        float rxy = length(v.xy);
        float theta = atan(rxy, -v.z);
        vec2 dir = rxy > 1e-8 ? v.xy / rxy : vec2(0.0);
        return focal * theta * dir;
    }
    // Brown-Conrady on the normalized image plane.
    float invZ = 1.0 / -v.z;
    vec2 n = v.xy * invZ;
    float r2 = dot(n, n);
    float radial = 1.0 + lensParams.x * r2 + lensParams.y * r2 * r2;
    vec2 tangential = vec2(
        2.0 * lensParams.z * n.x * n.y + lensParams.w * (r2 + 2.0 * n.x * n.x),
        lensParams.z * (r2 + 2.0 * n.y * n.y) + 2.0 * lensParams.w * n.x * n.y
    );
    return focal * (n * radial + tangential);
}

void main() {
    // Default to outside the frustum so it's discarded if we return early
    gl_Position = vec4(0.0, 0.0, 2.0, 1.0);

    ivec2 orderingCoord = ivec2((gl_InstanceID >> 2) & 4095, gl_InstanceID >> 14);
    uint splatIndex = texelFetch(ordering, orderingCoord, 0)[gl_InstanceID & 3];
    if (splatIndex == 0xffffffffu) {
        // Special value reserved for "no splat"
        return;
    }

    ivec3 texCoord = splatTexCoord(int(splatIndex));
    vec3 center, scales, xxyyzz, xyxzyz;
    vec4 quaternion, rgba;
    mat3 cov3D;
    bvec3 zeroScales = bvec3(false);

    if (enableExtSplats) {
        uvec4 ext1 = texelFetch(extSplats, texCoord, 0);
        float alpha = unpackSplatExtAlpha(ext1);
        if ((alpha == 0.0) || (alpha < minAlpha)) {
            return;
        }
        uvec4 ext2 = texelFetch(extSplats2, texCoord, 0);

        if (!enableCovSplats) {
            unpackSplatExt(ext1, ext2, center, scales, quaternion, rgba);
            zeroScales = equal(scales, vec3(0.0));
            if (all(zeroScales)) {
                return;
            }
        } else {
            unpackSplatExtCov(ext1, ext2, center, rgba, xxyyzz, xyxzyz);
            if (all(equal(xxyyzz, vec3(0.0))) && all(equal(xyxzyz, vec3(0.0)))) {
                return;
            }
        }
    } else {
        uvec4 packedData = texelFetch(extSplats, texCoord, 0);
        if (!enableCovSplats) {
            unpackSplatEncoding(packedData, center, scales, quaternion, rgba, vec4(0.0, 1.0, LN_SCALE_MIN, LN_SCALE_MAX));
            zeroScales = equal(scales, vec3(0.0));
            if (all(zeroScales)) {
                return;
            }
        } else {
            unpackSplatCovEncoding(packedData, center, rgba, xxyyzz, xyxzyz, vec4(0.0, 1.0, LN_SCALE_MIN, LN_SCALE_MAX));
            if (all(equal(xxyyzz, vec3(0.0))) && all(equal(xyxzyz, vec3(0.0)))) {
                return;
            }
        }

        rgba.a *= 2.0;
        if ((rgba.a == 0.0) || (rgba.a < minAlpha)) {
            return;
        }
    }

    // Match the reference 3DGS rasterizer (graphdeco-inria/diff-gaussian-rasterization),
    // which clamps the SH-evaluated color to positive
    rgba.rgb = max(rgba.rgb, vec3(0.0));

    adjustedStdDev = maxStdDev;
    if (rgba.a > 1.0) {
        // Stretch 1..2 to 1..5
        rgba.a = min(rgba.a * 4.0 - 3.0, 5.0);

        if (lodInflate) {
            // Adjust size to componsate for loss of opacity
            float opacity = exp((rgba.a * rgba.a - 1.0) / 2.718281828459045);
            float rescale = pow(opacity, 1.0 / 3.0);
            scales *= rescale;
            rgba.a = 1.0;
        }

        // Expand the maximum std dev to approximately cover the larger range
        adjustedStdDev = maxStdDev + 0.7 * (rgba.a - 1.0);
    }

    // Compute the view space center of the splat
    vec3 viewCenter = (!enableCovSplats ? quatVec(renderToViewQuat, center) : (renderToViewBasis * center)) + renderToViewPos;

    // Discard splats behind the camera
    if (viewCenter.z >= 0.0) {
        return;
    }

    // Compute the clip space center of the splat
    vec4 clipCenter = projectionMatrix * vec4(viewCenter, 1.0);

    // Discard splats outside near/far planes
    if (abs(clipCenter.z) >= clipCenter.w) {
        return;
    }

    // Discard splats more than clipXY times outside the XY frustum. A lens clips on its own map.
    float clip = clipXY * clipCenter.w;
    if (lensModel == 0 && (abs(clipCenter.x) > clip || abs(clipCenter.y) > clip)) {
        return;
    }

    vRgba = rgba;
    vSplatUv = position.xy * adjustedStdDev;

    // Record the splat index for entropy
    vSplatIndex = splatIndex;

    if (!enableCovSplats) {
        // Compute view space quaternion of splat
        vec4 viewQuaternion = quatQuat(renderToViewQuat, quaternion);

        if (enable2DGS && any(zeroScales)) {
            vec3 offset;
            if (zeroScales.z) {
                offset = vec3(vSplatUv.xy * scales.xy, 0.0);
            } else if (zeroScales.y) {
                offset = vec3(vSplatUv.x * scales.x, 0.0, vSplatUv.y * scales.z);
            } else {
                offset = vec3(0.0, vSplatUv.xy * scales.yz);
            }

            vec3 viewPos = viewCenter + quatVec(viewQuaternion, offset);
            gl_Position = projectionMatrix * vec4(viewPos, 1.0);
            vNdc = gl_Position.xyz / gl_Position.w;

            #include <logdepthbuf_vertex>
            return;
        }

        // Compute the 3D covariance matrix of the splat
        mat3 RS = scaleQuaternionToMatrix(scales, viewQuaternion);
        cov3D = RS * transpose(RS);
    } else {
        cov3D = mat3(
            xxyyzz.x, xyxzyz.x, xyxzyz.y,
            xyxzyz.x, xxyyzz.y, xyxzyz.z,
            xyxzyz.y, xyxzyz.z, xxyyzz.z
        );
        cov3D = renderToViewBasis * cov3D * transpose(renderToViewBasis);
    }

    // Compute the Jacobian of the splat's projection at its center
    vec2 scaledRenderSize = renderSize * focalAdjustment;
    vec2 focal = 0.5 * scaledRenderSize * vec2(projectionMatrix[0][0], projectionMatrix[1][1]);

    // Pixel centre of the splat, and its 2D covariance: from the EWA Jacobian for the pinhole,
    // from the unscented transform through the lens map otherwise.
    vec2 pixelCenter = vec2(0.0);
    mat3 cov2D;
    if (lensModel != 0 && !isOrthographic && !enableCovSplats) {
        // Sigma points along the splat's own axes, kappa = 0: centre +- sqrt(3) sigma, weight 1/6 each.
        mat3 RS = scaleQuaternionToMatrix(scales, quatQuat(renderToViewQuat, quaternion));
        vec2 p[6];
        vec2 mean = vec2(0.0);
        if (!lensValid(viewCenter)) {
            return;
        }
        for (int i = 0; i < 3; i++) {
            vec3 axis = 1.7320508 * RS[i];
            // A sigma point off the valid map stretches the splat across the frame: drop the splat.
            if (!lensValid(viewCenter + axis) || !lensValid(viewCenter - axis)) {
                return;
            }
            p[2 * i] = lensProject(viewCenter + axis, focal);
            p[2 * i + 1] = lensProject(viewCenter - axis, focal);
            mean += p[2 * i] + p[2 * i + 1];
        }
        mean /= 6.0;
        mat2 c = mat2(0.0);
        for (int i = 0; i < 6; i++) {
            vec2 d = p[i] - mean;
            c += (1.0 / 6.0) * mat2(d.x * d.x, d.x * d.y, d.x * d.y, d.y * d.y);
        }
        cov2D = mat3(c[0][0], c[0][1], 0.0, c[1][0], c[1][1], 0.0, 0.0, 0.0, 0.0);
        pixelCenter = mean;
        // The pinhole frustum clip above does not apply to a wide lens: clip on the mapped centre.
        if (any(greaterThan(abs(pixelCenter), clipXY * 0.5 * scaledRenderSize))) {
            return;
        }
    } else {
    mat3 J;
    if (isOrthographic) {
        J = mat3(
            focal.x, 0.0, 0.0,
            0.0, focal.y, 0.0,
            0.0, 0.0, 0.0
        );
    } else {
        float invZ = 1.0 / viewCenter.z;
        vec2 J1 = focal * invZ;
        vec2 J2 = -(J1 * viewCenter.xy) * invZ;
        J = mat3(
            J1.x, 0.0, J2.x,
            0.0, J1.y, J2.y,
            0.0, 0.0, 0.0
        );
    }

    // Compute the 2D covariance by projecting the 3D covariance
    // and picking out the XY plane components.
    cov2D = transpose(J) * cov3D * J;
    }
    float a = cov2D[0][0];
    float d = cov2D[1][1];
    float b = cov2D[0][1];

    // Optionally pre-blur the splat to match non-antialias optimized splats
    a += preBlurAmount;
    d += preBlurAmount;

    float fullBlurAmount = blurAmount;
    if ((focalDistance > 0.0) && (apertureAngle > 0.0)) {
        float focusRadius = maxPixelRadius;
        if (viewCenter.z < 0.0) {
            float focusBlur = abs((-viewCenter.z - focalDistance) / viewCenter.z);
            float apertureRadius = focal.x * tan(0.5 * apertureAngle);
            focusRadius = focusBlur * apertureRadius;
        }
        fullBlurAmount = clamp(sqr(focusRadius), blurAmount, sqr(maxPixelRadius));
    }

    // Do convolution with a 0.5-pixel Gaussian for anti-aliasing: sqrt(0.3) ~= 0.5
    float detOrig = a * d - b * b;
    a += fullBlurAmount;
    d += fullBlurAmount;
    float det = a * d - b * b;

    // Compute anti-aliasing intensity scaling factor
    float blurAdjust = sqrt(max(0.0, detOrig / det));
    rgba.a *= blurAdjust;
    if (rgba.a < minAlpha) {
        return;
    }
    vRgba.a = rgba.a;

    // Compute the eigenvalue and eigenvectors of the 2D covariance matrix
    float eigenAvg = 0.5 * (a + d);
    float eigenDelta = sqrt(max(0.0, eigenAvg * eigenAvg - det));
    float eigen1 = eigenAvg + eigenDelta;
    float eigen2 = eigenAvg - eigenDelta;

    vec2 eigenVec1 = (abs(b) > 0.001) ? normalize(vec2(b, eigen1 - a))
        : ((a >= d) ? vec2(1.0, 0.0) : vec2(0.0, 1.0));
    vec2 eigenVec2 = vec2(eigenVec1.y, -eigenVec1.x);

    float scale1 = min(maxPixelRadius, adjustedStdDev * sqrt(eigen1));
    float scale2 = min(maxPixelRadius, adjustedStdDev * sqrt(eigen2));
    if (scale1 < minPixelRadius && scale2 < minPixelRadius) {
        return;
    }

    // Compute the NDC coordinates for the ellipsoid's diagonal axes.
    vec2 pixelOffset = position.x * eigenVec1 * scale1 + position.y * eigenVec2 * scale2;
    vec2 ndcOffset = (2.0 / scaledRenderSize) * pixelOffset;

    // Compute NDC center of the splat
    vec3 ndcCenter = clipCenter.xyz / clipCenter.w;
    if (lensModel != 0 && !isOrthographic && !enableCovSplats) {
        ndcCenter.xy = pixelCenter * (2.0 / scaledRenderSize);
    }
    vec3 ndc = vec3(ndcCenter.xy + ndcOffset, ndcCenter.z);

    vNdc = ndc;
    gl_Position = vec4(ndc.xy * clipCenter.w, clipCenter.zw);

    #include <logdepthbuf_vertex>
}
