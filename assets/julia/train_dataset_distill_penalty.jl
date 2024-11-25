module M

include("common_defs.jl")
include("utilities.jl")

const λ_grad_penalty = 100.f0
const λ_inner_loss = 1.f0
const λ_divers = 1.f0

const η = 1f-4

########################################################################

# Function to be applied on the squared norm of the gradient
penalizer_fn(u) = sqrt(1.f0 + u) - 1.f0  # smooth l1
#penalizer_fn(u) = sqrt(1.f0 + u) - 1.f0 + u  # smooth l1-l2
#penalizer_fn = identity

# Auto-diff unfortunately doesn't work on this function
function _penalizer_term(m, Xs)
    gs = gradient(mm -> loss_AE(mm, Xs), m)[1]
    return penalizer_fn(sum(abs2.(gs[1])) + sum(abs2.(gs[2])) + sum(abs2.(gs[3])))
end

function penalizer_term(m, Xs)
    # Manual computation of the gradient
    as = m.W' * Xs .+ m.b
    hiddens = my_σ.(as)
    dhiddens = ∇my_σ.(as)

    XXs = m.W * hiddens .+ m.a / σ2_AE
    residual = (XXs - Xs / σ2_AE) / N
    W_r = m.W' * residual

    gs_W = residual * hiddens' + Xs * (W_r .* dhiddens)' + λ_weight_decay * m.W
    gs_b = sum(W_r .* dhiddens, dims=2)
    gs_a = sum(residual, dims=2) / σ2_AE

    return penalizer_fn(sum(abs2.(gs_W)) + sum(abs2.(gs_b)) + sum(abs2.(gs_a)))
end

########################################################################

function train_distilled_dataset()
    n_train = 10000
    batch_size = 100
    epochs = 200

    train_xs, train_ys = MNIST(split=:train)[:]
    train_Xs = reshape(train_xs, 28*28, :)[:, 1:n_train]
    train_Ys = onehotbatch(train_ys[1:n_train], 0:9)

    test_xs, test_ys = MNIST(split=:test)[:]
    test_Xs = reshape(test_xs, 28*28, :)
    test_Ys = onehotbatch(test_ys, 0:9)

    m_base = Model_AE(28*28 => K)
    m_head = Model_Head(K)

    Xs_distill0 = create_initial_Xs_distill()
    Xs_distill = deepcopy(Xs_distill0)
    save("Xs_distilled_init.png", render_filters(Xs_distill', N_x, N_y, 28, 28, individual_scaling = false, center_gray = false))

    # Pretrain AE (optional)
    println("Initial AE loss: $(loss_AE(m_base, Xs_distill))")
    #@time train_initial_AE!(m_base, Xs_distill)
    println("Intermediate AE loss: $(loss_AE(m_base, Xs_distill))")

    #XXs = apply_AE(m_base, Xs_distill)
    #save("W_base_init0.png", render_filters(m_base.W', K_x, K_y, 28, 28, individual_scaling = false, center_gray = true))
    #save("XXs_distilled_init.png", render_filters(XXs', N_x, N_y, 28, 28, individual_scaling = false, center_gray = false))

    # Check if the two losses return the same value
    @show _penalizer_term(m_base, Xs_distill)
    @show penalizer_term(m_base, Xs_distill)

    #return   # If only the AE pretraining is tested

    train_loader = DataLoader((data=train_Xs, label=train_Ys), batchsize=batch_size, partial=false, shuffle=true);

    ps = params(m_base.W, m_base.b, m_base.a, Xs_distill, m_head.W, m_head.b)

    opt = Adam(η)

    for epoch = 1:epochs
        total_loss = 0.0
        n_batches = 0

        for (xs_tr, ys_tr) in train_loader
            cur_loss, gs = withgradient(ps) do
                (classification_loss(apply_model(m_base, m_head, xs_tr), ys_tr)
                 + λ_grad_penalty * penalizer_term(m_base, Xs_distill)
                 + λ_inner_loss * loss_AE(m_base, Xs_distill)
                 - λ_divers * mse(Xs_distill, repeat(sum(Xs_distill, dims=2) / N, 1, N)))
            end
            update!(opt, ps, gs)
            total_loss += cur_loss
            n_batches += 1
        end # for (xs_val, ys_val)

        test_accuracy = accuracy(apply_model(m_base, m_head, test_Xs), test_Ys)
        total_loss /= n_batches

        # Output some statistics and visualizations
        save("W_base.png", render_filters(m_base.W', K_x, K_y, 28, 28, individual_scaling = false, center_gray = true))
        save("Xs_distilled.png", render_filters(Xs_distill', N_x, N_y, 28, 28, individual_scaling = false, center_gray = false))

        XXs = apply_AE(m_base, Xs_distill)
        save("XXs_distilled.png", render_filters(XXs', N_x, N_y, 28, 28, individual_scaling = false, center_gray = false))

        println("epoch: $epoch \t training loss: $total_loss \t test accuracy: $(100*test_accuracy)%",
                "\t |W_base|: $(norm(m_base.W)) \t |Xs|: $(norm(Xs_distill)) \t |Xs-XXs|: $(norm(XXs - Xs_distill))")
    end # for epoch
end

train_distilled_dataset()

end
