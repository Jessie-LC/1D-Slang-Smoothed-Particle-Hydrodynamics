import aperture;
import include.particles.buffer;
import include.particles.constants;
import include.kernels.normal;
import include.boundary.toroidal;

extern static const int PARTICLE_COUNT;
extern static const int PARTICLE_GROUP_SIZE;

struct ComputeInput {
    uint groupIndex : SV_GroupIndex;
    uint3 groupID : SV_GroupID;
    uint3 localID : SV_GroupThreadID;
    uint3 globalID : SV_DispatchThreadID;
};

float3 ColorDiffusion(in particle_t particle, in uint particleID) {
    precise float3 diffusion = float3(0.0f, 0.0f, 0.0f);
    for(uint i = 0u; i < PARTICLE_COUNT; ++i) {
        if (i == particleID) {
            continue;
        }

        particle_t found = particles[i];

        if (found.status == 0) {
            continue;
        }

        precise float32_t dx = ToroidalDifference(found.position, particle.position);
        precise float32_t dist = ToroidalDistance(found.position, particle.position);
        precise float32_t kernel = NormalKernel(dist, smoothingLength);

        precise float32_t bulkDensity = (found.density + particle.density) / 2.0;
        diffusion += (found.mass / bulkDensity) * (found.color - particle.color) * kernel;
    }

    return 1.0f * diffusion;
}

[[shader("compute")]]
[numthreads(PARTICLE_GROUP_SIZE, 1, 1)]
void IntegrateVelocity(ComputeInput input) {
    uint32_t particleID = input.globalID.x;

    particle_t particle = particles[particleID];

    particle.velocity = particle.velocity + particle.acceleration * 0.5f * deltaTime;
    particle.color = particle.color + ColorDiffusion(particle, particleID) * deltaTime * 0.5f;

    particles[particleID] = particle;
}