import aperture;
import include.particles.buffer;
import include.particles.constants;
import include.boundary.toroidal;
import include.kernels.normal;
import include.physics.constants;

extern static const int PARTICLE_COUNT;
extern static const int PARTICLE_GROUP_SIZE;

struct ComputeInput {
    uint groupIndex : SV_GroupIndex;
    uint3 groupID : SV_GroupID;
    uint3 localID : SV_GroupThreadID;
    uint3 globalID : SV_DispatchThreadID;
};

[[shader("compute")]]
[numthreads(PARTICLE_GROUP_SIZE, 1, 1)]
void CalculateDensity(ComputeInput input) {
    uint32_t particleID = input.globalID.x;

    particle_t particle = particles[particleID];

    precise float32_t density = float32_t(0.0f);
    for(int32_t i = 0; i < PARTICLE_COUNT; ++i) {
        particle_t found = particles[i];

        precise float32_t dist = ToroidalDistance(found.position, particle.position);
        precise float2 dx = ToroidalDifference(found.position, particle.position);
        precise float2 direction = dx / dist;
        if (dist == 0.0f) {
            direction = float2(0.0f, 0.0f);
        }

        density += found.mass * NormalKernel(dist, smoothingLength);
    }

    particle.density = density;

    particles[particleID] = particle;
}