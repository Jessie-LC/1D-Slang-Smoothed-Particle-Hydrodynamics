import aperture;
import include.particles.buffer;
import include.particles.constants;
import include.boundary.toroidal;
import include.kernels.normal;
import include.rendering.buffer;

extern static const int PARTICLE_COUNT;
extern static const int PARTICLE_GROUP_SIZE;
extern static const int RENDER_WIDTH;
extern static const int RENDER_HEIGHT;

struct ComputeInput {
    uint groupIndex : SV_GroupIndex;
    uint3 groupID : SV_GroupID;
    uint3 localID : SV_GroupThreadID;
    uint3 globalID : SV_DispatchThreadID;
};

// https://iquilezles.org/articles/palettes/
float3 palette(in float t, in float3 a, in float3 b, in float3 c, in float3 d) {
    return a + b*cos( 6.283185*(c*t+d) );
}

void DrawCircle(in particle_t particle) {
    int32_t particlePixelPosition = int32_t(((particle.position + boundLength / 2.0f) / boundLength) * float32_t(RENDER_WIDTH));
    int32_t sLength = int32_t((smoothingLength / boundLength) * RENDER_WIDTH);

    if (sLength == 0) {
        return;
    }

    for(int32_t x = -sLength; x <= sLength; ++x) {
        int32_t pixelCoordinate = (particlePixelPosition + x) % RENDER_WIDTH;

        float32_t coord = float32_t(pixelCoordinate.x) / float32_t(RENDER_WIDTH);
        coord = coord - float32_t((RENDER_WIDTH / 2.0f) / RENDER_WIDTH);

        float32_t spatialPosition = coord * boundLength;
        float32_t clampedPosition = clamp(spatialPosition, -boundLength/2.0f, boundLength/2.0f);

        if (spatialPosition != clampedPosition) {
            continue;
        }

        float32_t tmpPosition = particle.position;
        tmpPosition = tmpPosition + boundLength / 2.0f;
        tmpPosition = fmod(tmpPosition, boundLength);
        tmpPosition = tmpPosition - boundLength / 2.0f;

        float32_t dx = ToroidalDifference(spatialPosition, tmpPosition);
        float32_t dist = ToroidalDistance(spatialPosition, tmpPosition);

        float32_t centerKernel = NormalKernel(0.0f, smoothingLength);
        float32_t kernel = NormalKernel(dist, smoothingLength);
        float32_t reNormalizedKernel = kernel / centerKernel;

        float32_t alpha = reNormalizedKernel;

        float3 color;
        //float32_t t = exp(-(1.0f - (length(particle.velocity) / 100.0f)) * 1.4f);
        //color = palette(t,float3(1.0,0.4,0.0),float3(0.4,0.8,0.0),float3(0.5,0.3,0.9),float3(0.9,0.6,0.9));
        color = particle.color * particle.density * kernel;
        color = color * 1.0f;
        color = color / (1.0f + color);

        int32_t pixelIndex = pixelCoordinate;
        InterlockedMax<uint32_t>(pixelbuffer[pixelIndex].pixel, PackPixelColor(float4(color, alpha)));
    }
}

[[shader("compute")]]
[numthreads(PARTICLE_GROUP_SIZE, 1, 1)]
void DrawParticles(ComputeInput input) {
    uint32_t particleID = input.globalID.x;

    particle_t particle = particles[particleID];

    DrawCircle(particle);
}