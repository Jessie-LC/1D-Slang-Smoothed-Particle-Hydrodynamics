public module ToroidalDistance;

import include.particles.buffer;
import include.particles.constants;

float32_t ToroidalDistance(float32_t p1, float32_t p2) {
    float32_t dp = abs(p1 - p2);
    dp = lerp(dp, boundLength - dp, float32_t(dp > (boundLength / 2.0f)));
    return length(dp);
}
float32_t ToroidalDifference(float32_t p1, float32_t p2) {
    float32_t offs = p1 - p2;
    float32_t absOffs = abs(offs);
    return lerp(absOffs, absOffs - boundLength, float32_t(absOffs > (boundLength / 2.0f))) * sign(offs);
}