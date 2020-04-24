/**
 * \file dnn/src/aarch64/relayout/opr_impl.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#pragma once
#include "megdnn/oprs.h"
#include "src/fallback/relayout/opr_impl.h"

namespace megdnn {
namespace aarch64 {

class RelayoutForwardImpl final : public fallback::RelayoutForwardImpl {
 public:
    using fallback::RelayoutForwardImpl::RelayoutForwardImpl;

    void exec(_megdnn_tensor_in src, _megdnn_tensor_out dst,
              Handle *src_handle) override;

    bool is_thread_safe() const override { return true; }
};

}  // namespace aarch64
}  // namespace megdnn

// vim: syntax=cpp.doxygen
