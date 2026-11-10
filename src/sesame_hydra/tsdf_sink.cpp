// TsdfSink: a Hydra output sink that hands the TSDF (signed distance and weight per voxel) to Python.
//
// Hydra calls every active-window sink after each frame with the current volumetric map. This sink copies the
// distance and weight arrays of every block in the window into one process-wide store, keyed by block index, so
// blocks that later leave the window keep their last values. The same shared library is a Python module:
//
//   import hydra_python as hydra, sesame_hydra
//   hydra.init_plugins(ExternalPluginConfig(paths=[sesame_hydra.__file__]))   # registers "TsdfSink"
//   ... active_window: {sinks: [{type: TsdfSink}]} ... run the pipeline ...
//   voxels = sesame_hydra.tsdf_voxels()   # every observed voxel near a surface
//
// Nothing in Hydra is modified: the integrator's weights are read as they are.

#include <config_utilities/config.h>
#include <config_utilities/factory.h>
#include <hydra/active_window/active_window_module.h>
#include <hydra/active_window/active_window_output.h>
#include <hydra/reconstruction/volumetric_map.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <map>
#include <mutex>
#include <tuple>
#include <vector>

namespace sesame_hydra {

namespace py = pybind11;

struct TsdfStore {
  struct Block {
    std::vector<float> distance;
    std::vector<float> weight;
  };

  std::mutex mutex;
  float voxel_size = 0.0f;
  float truncation_distance = 0.0f;
  int voxels_per_side = 0;
  size_t num_calls = 0;
  std::map<std::tuple<int, int, int>, Block> blocks;

  static TsdfStore& instance() {
    static TsdfStore store;
    return store;
  }
};

struct TsdfSink : hydra::ActiveWindowModule::Sink {
  struct Config {};

  explicit TsdfSink(const Config&) {}

  std::string printInfo() const override { return "TsdfSink: copies TSDF distance and weight for Python"; }

  void call(uint64_t, const hydra::VolumetricMap& map, const hydra::ActiveWindowOutput&) const override {
    auto& store = TsdfStore::instance();
    std::lock_guard<std::mutex> lock(store.mutex);
    store.voxel_size = map.config.voxel_size;
    store.truncation_distance = map.config.truncation_distance;
    store.voxels_per_side = static_cast<int>(map.config.voxels_per_side);
    ++store.num_calls;
    for (const auto& block : map.getTsdfLayer()) {
      auto& copy = store.blocks[{block.index.x(), block.index.y(), block.index.z()}];
      const size_t n = block.numVoxels();
      copy.distance.resize(n);
      copy.weight.resize(n);
      for (size_t i = 0; i < n; ++i) {
        const auto& voxel = block.getVoxel(i);
        copy.distance[i] = voxel.distance;
        copy.weight[i] = voxel.weight;
      }
    }
  }
};

void declare_config(TsdfSink::Config&) { config::name("TsdfSink::Config"); }

namespace {
static const auto registration =
    config::RegistrationWithConfig<hydra::ActiveWindowModule::Sink, TsdfSink, TsdfSink::Config>("TsdfSink");
}  // namespace

// Every voxel with weight > 0 and |distance| < truncation, as numpy arrays. Global voxel index g has its centre at
// (g + 0.5) * voxel_size (spatial_hash convention), so the lattice origin is 0.5 * voxel_size on every axis.
py::dict tsdf_voxels() {
  auto& store = TsdfStore::instance();
  std::lock_guard<std::mutex> lock(store.mutex);
  std::vector<int32_t> index;
  std::vector<float> distance, weight;
  const int n = store.voxels_per_side;
  for (const auto& [key, block] : store.blocks) {
    const auto [bx, by, bz] = key;
    for (size_t i = 0; i < block.weight.size(); ++i) {
      if (!(block.weight[i] > 0.0f) || std::abs(block.distance[i]) >= store.truncation_distance) {
        continue;
      }
      // spatial_hash linear index: x fastest, then y, then z
      const int lx = static_cast<int>(i % n), ly = static_cast<int>((i / n) % n), lz = static_cast<int>(i / (n * n));
      index.insert(index.end(), {bx * n + lx, by * n + ly, bz * n + lz});
      distance.push_back(block.distance[i]);
      weight.push_back(block.weight[i]);
    }
  }
  const auto count = static_cast<py::ssize_t>(weight.size());
  py::array_t<int32_t> index_array({count, static_cast<py::ssize_t>(3)});
  std::copy(index.begin(), index.end(), index_array.mutable_data());
  py::array_t<float> distance_array(count), weight_array(count);
  std::copy(distance.begin(), distance.end(), distance_array.mutable_data());
  std::copy(weight.begin(), weight.end(), weight_array.mutable_data());
  py::dict out;
  out["index"] = index_array;
  out["distance"] = distance_array;
  out["weight"] = weight_array;
  out["voxel_size"] = store.voxel_size;
  out["origin"] = 0.5f * store.voxel_size;
  out["truncation_distance"] = store.truncation_distance;
  out["num_calls"] = store.num_calls;
  return out;
}

void clear() {
  auto& store = TsdfStore::instance();
  std::lock_guard<std::mutex> lock(store.mutex);
  store.blocks.clear();
  store.num_calls = 0;
}

PYBIND11_MODULE(sesame_hydra, m) {
  m.doc() = "TSDF voxels from Hydra's active window (see tsdf_sink.cpp)";
  m.def("tsdf_voxels", &tsdf_voxels, "every observed voxel near a surface: index (N,3), distance, weight");
  m.def("clear", &clear, "forget every stored block");
}

}  // namespace sesame_hydra
