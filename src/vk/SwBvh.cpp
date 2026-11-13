// CalculatoRTX - construction BVH2 par SAH pour le mode ray tracing logiciel.
#include "SwBvh.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>

namespace crtx {

namespace {

constexpr int kBins = 8;

struct Aabb {
    float3 bmin = make_float3(1e30f, 1e30f, 1e30f);
    float3 bmax = make_float3(-1e30f, -1e30f, -1e30f);

    void grow(float3 p)
    {
        bmin = make_float3(std::fmin(bmin.x, p.x), std::fmin(bmin.y, p.y), std::fmin(bmin.z, p.z));
        bmax = make_float3(std::fmax(bmax.x, p.x), std::fmax(bmax.y, p.y), std::fmax(bmax.z, p.z));
    }

    void grow(const Aabb& o)
    {
        bmin = make_float3(std::fmin(bmin.x, o.bmin.x), std::fmin(bmin.y, o.bmin.y), std::fmin(bmin.z, o.bmin.z));
        bmax = make_float3(std::fmax(bmax.x, o.bmax.x), std::fmax(bmax.y, o.bmax.y), std::fmax(bmax.z, o.bmax.z));
    }

    float area() const
    {
        const float3 e = bmax - bmin;
        if (e.x < 0.0f || e.y < 0.0f || e.z < 0.0f) return 0.0f;
        return 2.0f * (e.x * e.y + e.y * e.z + e.z * e.x);
    }
};

float comp(float3 v, int axis) { return axis == 0 ? v.x : (axis == 1 ? v.y : v.z); }

void subdivide(uint32_t nodeIdx, uint32_t start, uint32_t count, const std::vector<BvhPrim>& prims,
               std::vector<BvhNode>& nodes, std::vector<uint32_t>& indices, uint32_t leafMax)
{
    Aabb box, cbox;
    for (uint32_t i = 0; i < count; ++i) {
        const BvhPrim& p = prims[indices[start + i]];
        box.grow(p.bmin);
        box.grow(p.bmax);
        cbox.grow(p.centroid);
    }
    nodes[nodeIdx].minX = box.bmin.x;
    nodes[nodeIdx].minY = box.bmin.y;
    nodes[nodeIdx].minZ = box.bmin.z;
    nodes[nodeIdx].maxX = box.bmax.x;
    nodes[nodeIdx].maxY = box.bmax.y;
    nodes[nodeIdx].maxZ = box.bmax.z;

    if (count <= leafMax) {
        nodes[nodeIdx].leftFirst = start;
        nodes[nodeIdx].count = count;
        return;
    }

    // Recherche de la meilleure partition SAH sur les 3 axes
    int bestAxis = -1;
    float bestPos = 0.0f;
    float bestCost = 1e30f;
    const float parentArea = std::fmax(box.area(), 1e-12f);

    for (int axis = 0; axis < 3; ++axis) {
        const float cmin = comp(cbox.bmin, axis);
        const float cmax = comp(cbox.bmax, axis);
        const float extent = cmax - cmin;
        if (extent <= 1e-7f) continue;

        struct Bin {
            Aabb box;
            uint32_t count = 0;
        } bins[kBins];

        const float scale = static_cast<float>(kBins) / extent;
        for (uint32_t i = 0; i < count; ++i) {
            const BvhPrim& p = prims[indices[start + i]];
            const int b = std::clamp(static_cast<int>((comp(p.centroid, axis) - cmin) * scale), 0, kBins - 1);
            bins[b].count++;
            bins[b].box.grow(p.bmin);
            bins[b].box.grow(p.bmax);
        }

        float leftArea[kBins - 1], rightArea[kBins - 1];
        uint32_t leftCount[kBins - 1], rightCount[kBins - 1];
        Aabb lbox, rbox;
        uint32_t lcnt = 0, rcnt = 0;
        for (int i = 0; i < kBins - 1; ++i) {
            lcnt += bins[i].count;
            lbox.grow(bins[i].box);
            leftCount[i] = lcnt;
            leftArea[i] = lbox.area();

            const int j = kBins - 1 - i;
            rcnt += bins[j].count;
            rbox.grow(bins[j].box);
            rightCount[j - 1] = rcnt;
            rightArea[j - 1] = rbox.area();
        }

        const float invScale = extent / static_cast<float>(kBins);
        for (int i = 0; i < kBins - 1; ++i) {
            if (leftCount[i] == 0 || rightCount[i] == 0) continue;
            const float cost = 1.0f + (leftCount[i] * leftArea[i] + rightCount[i] * rightArea[i]) / parentArea;
            if (cost < bestCost) {
                bestCost = cost;
                bestAxis = axis;
                bestPos = cmin + (i + 1) * invScale;
            }
        }
    }

    uint32_t mid = start;
    if (bestAxis >= 0) {
        auto it = std::partition(indices.begin() + start, indices.begin() + start + count,
                                 [&](uint32_t idx) { return comp(prims[idx].centroid, bestAxis) < bestPos; });
        mid = static_cast<uint32_t>(it - indices.begin());
    }
    if (mid == start || mid == start + count) {
        // Repli médian si toutes les primitives ont le même centroïde ou partition vide
        const float3 e = cbox.bmax - cbox.bmin;
        const int axis = e.x >= e.y && e.x >= e.z ? 0 : (e.y >= e.z ? 1 : 2);
        mid = start + count / 2;
        std::nth_element(indices.begin() + start, indices.begin() + mid, indices.begin() + start + count,
                         [&](uint32_t a, uint32_t b) { return comp(prims[a].centroid, axis) < comp(prims[b].centroid, axis); });
    }

    const uint32_t leftIdx = static_cast<uint32_t>(nodes.size());
    nodes.resize(leftIdx + 2);
    nodes[nodeIdx].leftFirst = leftIdx;
    nodes[nodeIdx].count = 0;

    subdivide(leftIdx, start, mid - start, prims, nodes, indices, leafMax);
    subdivide(leftIdx + 1, mid, start + count - mid, prims, nodes, indices, leafMax);
}

}  // namespace

void buildBvh(const std::vector<BvhPrim>& prims, std::vector<BvhNode>& nodes, std::vector<uint32_t>& indices,
              uint32_t leafMax)
{
    nodes.clear();
    indices.resize(prims.size());
    std::iota(indices.begin(), indices.end(), 0u);
    if (prims.empty()) return;
    nodes.reserve(prims.size() * 2);
    nodes.emplace_back();
    subdivide(0, 0, static_cast<uint32_t>(prims.size()), prims, nodes, indices, leafMax);
}

Affine invertAffine(const Affine& a)
{
    const float* m = a.m;
    const float c00 = m[5] * m[10] - m[6] * m[9];
    const float c01 = m[6] * m[8] - m[4] * m[10];
    const float c02 = m[4] * m[9] - m[5] * m[8];
    const float det = m[0] * c00 + m[1] * c01 + m[2] * c02;
    const float invDet = std::fabs(det) > 1e-12f ? 1.0f / det : 0.0f;

    Affine inv{};
    inv.m[0] = c00 * invDet;
    inv.m[1] = (m[2] * m[9] - m[1] * m[10]) * invDet;
    inv.m[2] = (m[1] * m[6] - m[2] * m[5]) * invDet;
    inv.m[4] = c01 * invDet;
    inv.m[5] = (m[0] * m[10] - m[2] * m[8]) * invDet;
    inv.m[6] = (m[2] * m[4] - m[0] * m[6]) * invDet;
    inv.m[8] = c02 * invDet;
    inv.m[9] = (m[1] * m[8] - m[0] * m[9]) * invDet;
    inv.m[10] = (m[0] * m[5] - m[1] * m[4]) * invDet;

    const float3 t = make_float3(m[3], m[7], m[11]);
    const float3 it = xformVector(inv.m, t);
    inv.m[3] = -it.x;
    inv.m[7] = -it.y;
    inv.m[11] = -it.z;
    return inv;
}

void transformAabb(const Affine& xf, float3 bmin, float3 bmax, float3& outMin, float3& outMax)
{
    outMin = make_float3(1e30f, 1e30f, 1e30f);
    outMax = make_float3(-1e30f, -1e30f, -1e30f);
    for (int i = 0; i < 8; ++i) {
        const float3 c = make_float3((i & 1) ? bmax.x : bmin.x, (i & 2) ? bmax.y : bmin.y, (i & 4) ? bmax.z : bmin.z);
        const float3 w = xformPoint(xf.m, c);
        outMin = make_float3(std::fmin(outMin.x, w.x), std::fmin(outMin.y, w.y), std::fmin(outMin.z, w.z));
        outMax = make_float3(std::fmax(outMax.x, w.x), std::fmax(outMax.y, w.y), std::fmax(outMax.z, w.z));
    }
}

}  // namespace crtx

