public module ParticleBuffer;

struct Particle {
    float3 color;

    float32_t position;

    float32_t velocity;
    float32_t acceleration;

    float32_t mass;
    float32_t density;
    float32_t gradH;

    int32_t status;
};

typedef Particle particle_t;

RWStructuredBuffer<particle_t> particles;