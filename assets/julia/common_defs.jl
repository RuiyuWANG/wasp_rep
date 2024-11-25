using Flux, Flux.Optimise
using Flux: onehotbatch, onecold
using Flux: logitcrossentropy, mse, params
using Zygote: ignore, withgradient
using MLUtils
using MLDatasets

import Random
import Statistics

########################################################################

const K_x, K_y = 8, 8; const K = K_x*K_y  # Number of features
const N_x, N_y = 20, 15; const N = N_x*N_y  # Size of synthetic dataset

const λ_weight_decay = 1f-4
const σ2_AE = 1.f0

struct Model_AE
    W :: Matrix{Float32}
    b :: Vector{Float32}
    a :: Vector{Float32}
end

struct Model_Head
    W :: Matrix{Float32}
    b :: Vector{Float32}
end

function Model_AE((dim_in, dim_out)::Pair{<:Integer, <:Integer})
    return Model_AE(Flux.glorot_uniform(dim_in, dim_out), zeros(Float32, dim_out), zeros(Float32, dim_in))
end

function Model_Head(dim_in :: Int)
    return Model_Head(Flux.glorot_uniform(dim_in, 10), zeros(Float32, 10))
end

#my_σ = sigmoid_fast; ∇my_σ(x) = sigmoid_fast(x) * (1.f0 - sigmoid_fast(x))
my_σ = relu; ∇my_σ(x) = ifelse(x > 0.f0, 1.f0, 0.f0)
#my_σ = leakyrelu; ∇my_σ(x) = ifelse(x > 0.f0, 1.f0, 0.01f0)
#my_σ = identity; ∇my_σ(x) = 1.f0

#∇my_σ(x) = gradient(my_σ, x)[1] # Turns out to be slow!

Random.seed!(42)

accuracy(x, y) = sum(onecold(x, 1:10) .== onecold(y, 1:10)) / size(y, 2)

function classification_loss(ys_pred, ys_gt)
    return logitcrossentropy(ys_pred, ys_gt)
    #return mse(ys_pred, ys_gt)
end

function apply_AE(m, Xs)
    return m.W * my_σ.(m.W' * Xs .+ m.b) .+ m.a / σ2_AE
end

function apply_model(m_base, m_head, xs)
    zs = my_σ.(m_base.W' * xs .+ m_base.b)
    return m_head.W' * zs .+ m_head.b
end

function loss_AE(m, Xs)
    B = size(Xs, 2)
    XXs = apply_AE(m, Xs)
    ΔXs = XXs - Xs / σ2_AE
    return 0.5f0 * (sum(abs2.(ΔXs)) / B + λ_weight_decay * sum(abs2.(m.W)))
end

params_base(m) = params((m.W, m.b, m.a))

function train_initial_AE!(m, Xs)
    epochs_inner = 250
    opt = Adam(1f-3)
    ps  = params_base(m)

    for epoch = 0:epochs_inner
        for xs in BatchView(Xs, batchsize=N, partial=false)
            gs = gradient(ps) do
                loss_AE(m, xs)
            end
            update!(opt, ps, gs)
        end # for xs
        if epoch % 25 == 0
            println("epoch: $epoch  AE loss: $(loss_AE(m, Xs))  |W|: $(norm(m.W))")
        end
    end # for epoch
end

function create_initial_Xs_distill()
    #Xs_init = reshape(MNIST(split=:train).features, 28*28, :)[:, 1:N]
    #Xs_init = reshape(FashionMNIST(split=:train).features, 28*28, :)[:, 1:N]
    #Xs_init = reshape(EMNIST(:letters, split=:train).features, 28*28, :)[:, 1:N]

    Xs_distill = rand(Float32, 28*28, N)
    #Xs_distill .= 0.25f0 * Xs_init + 0.75f0 * rand(Float32, 28*28, N)
    #Xs_distill = Xs_init .* rand(Float32, 28*28, N)
    Xs_distill ./= 16

    return Xs_distill
end
