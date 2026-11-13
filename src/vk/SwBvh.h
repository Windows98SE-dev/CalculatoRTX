// CalculatoRTX - construction BVH2 sur CPU pour le mode ray tracing logiciel (low-end).
#pragma once

#include "../common/VecMath.h"

#include <cstdint>
#include <vector>

namespace crtx {

// Nœud BVH2 de 32 octets (disposition identique à BvhNode dans Common.glsl).
struct BvhNode {
    float minX, minY, minZ;
    uint32_t leftFirst;  // si count > 0 : 1er indice dans indices ; sinon : fils gauche (droit = leftFirst + 1)
    float maxX, maxY, maxZ;
    uint32_t count;      // 0 = nœud interne, > 0 = feuille
};
static_assert(sizeof(BvhNode) == 32, "BvhNode : disposition partagée avec les shaders");

// Instance pour la traversée logicielle (disposition identique à SwInstance dans Common.glsl).
struct SwInstance {
    float objectToWorld[12];
    float worldToObject[12];
    uint32_t instanceId;
    uint32_t mask;
    uint32_t geometry;
    uint32_t pad0;
};
static_assert(sizeof(SwInstance) == 112, "SwInstance : disposition partagée avec les shaders");

struct BvhPrim {
    float3 bmin;
    float3 bmax;
    float3 centroid;
};

// Construit un BVH2 par SAH (8 intervalles) sur les primitives données.
void buildBvh(const std::vector<BvhPrim>& prims, std::vector<BvhNode>& nodes, std::vector<uint32_t>& indices,
              uint32_t leafMax = 2);

// Calcule l'inverse d'une transformation affine 3x4.
Affine invertAffine(const Affine& a);

// Transforme une boîte englobante locale en boîte englobante monde.
void transformAabb(const Affine& xf, float3 bmin, float3 bmax, float3& outMin, float3& outMax);

}  // namespace crtx

