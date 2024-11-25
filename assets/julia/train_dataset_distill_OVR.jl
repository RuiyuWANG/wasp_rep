module M

include("common_defs.jl")
include("utilities.jl")

const β = 1.f0 / 128
const η = 1f-4

########################################################################

function train_AE!(m, Xs, epochs_inner = 50)
    opt = Adam(1f-4)
    ps  = params_base(m)

    for epoch = 1:epochs_inner
        gs = gradient(ps) do
            loss_AE(m, Xs)
        end
        update!(opt, ps, gs)
    end # for epoch
end

########################################################################

function train_distilled_dataset()
    n_train = 10000
    n_validation = 5000
    batch_size = 100
    epochs = 200

    train_xs, train_ys = MNIST(split=:train)[:]
    train_Xs = reshape(train_xs, 28*28, :)[:, 1:n_train]
    train_Ys = onehotbatch(train_ys[1:n_train], 0:9)

    test_xs, test_ys = MNIST(split=:test)[:]
    test_Xs = reshape(test_xs, 28*28, :)
    test_Ys = onehotbatch(test_ys, 0:9)

    val_Xs = test_Xs[:, 1:n_validation]
    val_Ys = test_Ys[:, 1:n_validation]

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

    @time train_AE!(m_base, Xs_distill, 1000)
    println("AE loss: $(loss_AE(m_base, Xs_distill)) \t |W|: $(norm(m_base.W))")

    #XXs = apply_AE(m_base, Xs_distill)
    #save("W_base_init.png", render_filters(m_base.W', K_x, K_y, 28, 28, individual_scaling = false, center_gray = true))
    #save("XXs_distilled_init.png", render_filters(XXs', N_x, N_y, 28, 28, individual_scaling = false, center_gray = false))

    #return   # If only the AE pretraining is tested

    m_base0 = deepcopy(m_base)

    train_loader = DataLoader((data=train_Xs, label=train_Ys), batchsize=batch_size, partial=false, shuffle=true);

    ps = params(m_head.W, m_head.b, m_base.W, m_base.b, m_base.a, m_base0.W, m_base0.b, m_base0.a, Xs_distill)

    opt = Adam(η)

    for epoch = 1:epochs
        # Further optimize m_base0
        loss_base       = loss_AE(m_base, Xs_distill)
        loss_base0_init = loss_AE(m_base0, Xs_distill)
        
        train_AE!(m_base0, Xs_distill, 25)
        loss_base0_new = loss_AE(m_base0, Xs_distill)

        println("loss_AE: $loss_base \t init. loss_AE0: $loss_base0_init \t new loss_AE0: $loss_base0_new")
        if loss_base0_new > loss_base
            # reset m_base0 to m_base if we are stuck in a poor basin
            m_base0.W .= m_base.W; m_base0.b .= m_base.b; m_base0.a .= m_base.a
        end

        total_loss = 0.0
        n_batches = 0
        for (xs_tr, ys_tr) in train_loader
            cur_loss, gs = withgradient(ps) do
                classification_loss(apply_model(m_base, m_head, xs_tr), ys_tr) + (loss_AE(m_base, Xs_distill) - loss_AE(m_base0, Xs_distill)) / β
            end

            # Flip sign for parameters that are maximized
            gs[m_base0.W] .*= -1.f0; gs[m_base0.b] .*= -1.f0; gs[m_base0.a] .*= -1.f0

            update!(opt, ps, gs)
            total_loss += cur_loss
            n_batches += 1
        end # for (xs_val, ys_val)

        test_accuracy0 = accuracy(apply_model(m_base0, m_head, test_Xs), test_Ys)
        test_accuracy  = accuracy(apply_model(m_base , m_head, test_Xs), test_Ys)
        total_loss /= n_batches

        # Output some statistics and visualizations
        save("W_base.png", render_filters(m_base.W', K_x, K_y, 28, 28, individual_scaling = false, center_gray = true))
        save("W_base0.png", render_filters(m_base0.W', K_x, K_y, 28, 28, individual_scaling = false, center_gray = true))
        save("Xs_distilled.png", render_filters(Xs_distill', N_x, N_y, 28, 28, individual_scaling = false, center_gray = false))

        XXs = apply_AE(m_base0, Xs_distill)
        save("XXs_distilled.png", render_filters(XXs', N_x, N_y, 28, 28, individual_scaling = false, center_gray = false))

        println("epoch: $epoch \t training loss: $total_loss \t test accuracy: $(100*test_accuracy0)% / $(100*test_accuracy)%",
                " \t |W_base|: $(norm(m_base.W)) \t |W_base0|: $(norm(m_base0.W)) \t |Xs|: $(norm(Xs_distill))",
                " \t |Xs-XXs|: $(norm(XXs - Xs_distill))")
    end # for epoch
end

train_distilled_dataset()

end
