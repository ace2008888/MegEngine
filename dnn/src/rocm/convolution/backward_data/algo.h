/**
 * \file dnn/src/rocm/convolution/backward_data/algo.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#pragma once

#include "src/rocm/convolution/helper.h"

namespace megdnn {
namespace rocm {

/*!
 * \brief base class for convolution algos
 *
 */
class ConvolutionBackwardDataImpl::AlgoBase : public Algorithm {
protected:
    ~AlgoBase() = default;

public:
    struct SizeArgs {
        HandleImpl* handle;
        CanonizedFilterMeta filter_meta;
        const TensorLayout *diff_layout, *grad_layout;
        ConvolutionBackwardDataImpl* opr;

        std::string to_string() const;
        convolution::MIOpenCacheKey to_miopen_algo_cache_key() const;
        void init_desc(convolution::MIOpenBwdDataDescs& desc) const {
            desc.set(filter_meta, *diff_layout, *grad_layout, opr->param());
        }
        SizeArgs(ConvolutionBackwardDataImpl* opr, const TensorLayout& filter,
                 const TensorLayout& diff, const TensorLayout& grad);
        SizeArgs(ConvolutionBackwardDataImpl* opr,
                 const CanonizedFilterMeta& filter, const TensorLayout& diff,
                 const TensorLayout& grad);

        convolution::ForwardSizeArgs as_fwd_args() const {
            return {handle, grad_layout, filter_meta, diff_layout};
        }
    };
    struct ExecArgs : public SizeArgs {
        const TensorND *filter_tensor, *diff_tensor, *grad_tensor;
        Workspace workspace;

        ExecArgs(ConvolutionBackwardDataImpl* opr, _megdnn_tensor_in filter,
                 _megdnn_tensor_in diff, _megdnn_tensor_out grad,
                 _megdnn_workspace workspace);
    };
    virtual bool is_available(const SizeArgs& args) const = 0;
    virtual size_t get_workspace_in_bytes(const SizeArgs& args) const = 0;
    virtual void exec(const ExecArgs& args) const = 0;

    bool is_available_wk(const SizeArgs& args, size_t limit) {
        return is_available(args) && get_workspace_in_bytes(args) <= limit;
    }
    bool is_available_reproducible(
            const SizeArgs& args, bool reproducible = true,
            size_t limit = std::numeric_limits<size_t>::max()) {
        return (!reproducible || is_reproducible()) &&
               is_available_wk(args, limit);
    }

    AlgoBase& check_workspace(const SizeArgs& args,
                              const Workspace& workspace) {
        auto req = get_workspace_in_bytes(args);
        megdnn_assert(req <= workspace.size,
                      "conv bwd data algo %s: "
                      "required workspace %zu bytes, got %zu",
                      name(), req, workspace.size);
        return *this;
    }

    virtual bool is_miopen() const { return false; }
};

class ConvolutionBackwardDataImpl::AlgoMIOpen final : public AlgoBase {
    bool m_is_reproducible;
    const char* m_name;

    miopenConvBwdDataAlgorithm_t find_best_algo(const ExecArgs& args);

public:
    AlgoMIOpen() = delete;
    AlgoMIOpen(bool is_reproducible) : m_is_reproducible(is_reproducible) {}

    bool is_available(const SizeArgs& args) const override;
    size_t get_workspace_in_bytes(const SizeArgs& args) const override;
    void exec(const ExecArgs& args) const override;

    bool is_reproducible() const override { return m_is_reproducible; }

    const char* name() const override {
        return "MIOpenConvolutionBackwardData";
    }

    bool is_miopen() const override { return true; }
    static convolution::MIOpenCache<SizeArgs, miopenConvBwdDataAlgorithm_t>
            sm_miopen_algo_cache;
    static convolution::MIOpenCache<SizeArgs, size_t> sm_miopen_ws_cache;
};

class ConvolutionBackwardDataImpl::AlgoMatmul final : public AlgoBase {
    template <typename T>
    static void exec_internal(const ExecArgs& args);

public:
    bool is_available(const SizeArgs& args) const override;
    size_t get_workspace_in_bytes(const SizeArgs& args) const override;
    void exec(const ExecArgs& args) const override;

    const char* name() const override { return "MATMUL"; }
    bool is_reproducible() const override { return true; }
};

class ConvolutionBackwardDataImpl::AlgoChanwise final : public AlgoBase {
public:
    bool is_available(const SizeArgs& args) const override;
    size_t get_workspace_in_bytes(const SizeArgs& args) const override;
    void exec(const ExecArgs& args) const override;

    const char* name() const override { return "CHANNEL_WISE"; }
    bool is_reproducible() const override { return true; }
};

class ConvolutionBackwardDataImpl::AlgoPack {
    // defined in miopen.cpp
    void fill_miopen_algos();

    AlgoPack(const AlgoPack&) = delete;
    AlgoPack& operator=(const AlgoPack&) = delete;

public:
    AlgoPack();

    AlgoMIOpen miopen{true};
    AlgoMatmul matmul;
    AlgoChanwise chanwise;

    std::vector<AlgoBase*>
            //! all algorithms
            all_algos, miopen_algos, non_miopen_algos;
};

} // namespace rocm
} // namespace megdnn

// vim: syntax=cpp.doxygen
