using Flux

function subgradient_training_flux(params_dict,training_dict,dict_all,training_loader)


    # Define your custom subgradient function
    function subgradient_spoplus(c_hat,l_p,l_s,params_dict)
        c = -l_p
        c_tilde = 2*c_hat-c
        c_tilde = reshape(c_tilde,1,length(c_tilde))

        w = calc_optimal_schedule(-c_tilde,params_dict)
        w = reshape(w,length(w),1)

        sg = 2 .*(w-reshape(l_s,length(l_s),1))

        return sg[1:params_dict["lookahead"]]

    end

    function get_activation_function(act_fct::String)

        if act_fct == "relu"
            return relu
        elseif act_fct == "softplus"
            return softplus
        elseif act_fct == "sigmoid"
            return sigmoid
        else
            exit("Invalid activation function string")
        end
    end

    function initialize_forecaster(params_dict,training_dict)
        ndf = params_dict["n_diff_features"]
        pos_fc = params_dict["pos_fc"]
        la = params_dict["lookahead"]
        ws = training_dict["warm_start"]

        @show(ws)

        model=nothing
        Random.seed!(73)

        if training_dict["train_mode"] == "linear"

            model = Chain(
                Dense(ndf*la,la, init= (dims...) -> -rand(la,ndf*la), bias=false)
                )

            if ws
                init = zeros(la,ndf*la)
                for i in 1:la
                    init[i,(i-1)*ndf+pos_fc] = -1
                end
                model.layers[1].weight .= init
            end
        
        elseif training_dict["train_mode"] == "nonlinear"

            act = get_activation_function(training_dict["activation_function"])

            model = Chain(
                Dense(ndf*la,la,act,init=(dims...) -> rand(la,ndf*la),bias=true),
                Dense(la,la,init=(dims...) -> -rand(la,la),bias=true)
            )

            if ws
                model = training_dict["pretrained_model"]
            end

        else
            error("Invalid train mode")
        end


        return model
    end

    function l1_loss_grad(p, λ)
        #Gradient of L1 regularization only applied to weights of neural network
        # return map(Flux.params(model)) do p
        #     if ndims(p) == 2  # It's likely a weight matrix
        #         return λ * sign.(p)
        #     else  # It's likely a bias vector or something else
        #         return zeros(size(p))
        #     end
        # end

        if ndims(p) == 2 #weights
            return λ*sign.(p)
        else #bias
            return zero(p)
        end
    end

    function get_fc_list_from_model(model)
        list = []

        for lyr in model.layers
            w = deepcopy(lyr.weight)
            push!(list,w)
            if hasproperty(lyr,:bias)
                b = deepcopy(lyr.bias)
                push!(list,b)
            end
        end

        return list
    end

    model = initialize_forecaster(params_dict,training_dict)
    list_fc_lists = [get_fc_list_from_model(model)]

    clip_val = training_dict["SG_clip"]

    n_examples = length(training_loader)*training_loader.batchsize


    λ = params_dict["reg"]

    # Training loop
    for e in 1:training_dict["SG_epochs"]

        lr = training_dict["lr"]/sqrt(e)
        println("Epoch: $(e), learning rate: $(lr)")

        ps = Flux.params(model)  # Get parameters
        total_grads = Dict()  # Initialize total_grads dictionary



        b=1

        for (f_batch, p_batch, s_batch) in training_loader
            # Initialize total_grads with zeros
            for (i,p) in enumerate(ps)
                total_grads[i] = zero(p)
            end
            
            features, labels_price, labels_schedule = transpose(f_batch), transpose(p_batch),transpose(s_batch)

            for i in 1:size(features)[1] #Stochastic gradient descent
                # Calculate subgradient with respect to output
                c_hat = model(features[i,:])

                subgrad_output = subgradient_spoplus(c_hat,labels_price[i,:],labels_schedule[i,:],params_dict)

                #subgrad_output -= weight_L2 * 2 .*(c_hat-labels_price[i,:])
        
                # Initialize total gradients by calculating regularization part 
                #total_grads = l1_loss_grad(model, λ) 

                # Combine subgradients and gradients
                for j in 1:length(c_hat)
                    #Calculate gradient of output c wrt model parameters
                    output_grads = gradient(() -> model(features[i,:])[j], ps)
        
                    for (i,g) in enumerate(output_grads)
                        if g !== nothing
                            total_grads[i] .+= subgrad_output[j] .* g
                        else
                            @warn "Gradient is Nothing. Skipping update for this parameter."
                        end
                    end
                end


                # Update step
                


            end

            for (i,p) in enumerate(Flux.params(model))
                @show(sum(abs.(total_grads[i])))
                grad_reg = l1_loss_grad(p,λ)

                grad_tot = -total_grads[i]/n_examples + grad_reg
                Flux.clamp!(grad_tot,-clip_val,clip_val)
                @show(sum(abs.(grad_tot)))


                Flux.Optimise.update!(p, lr .* grad_tot)
            end



            println("Updated batch $(b) in epoch $(e)")
            b+=1

        end       

        fc_list_epoch = get_fc_list_from_model(model)
        push!(list_fc_lists,fc_list_epoch)

    end

    return list_fc_lists

end


##### OLD FUNCTIONS #####

function subgradient_training(params_dict,training_dict,dict_all,training_loader)

    function calc_subgrad_spo(f,l_p,l_s,B)

        
        c_tilde = 2B*f - l_p
        c_tilde = reshape(c_tilde,1,length(c_tilde))


        w = calc_optimal_schedule(-c_tilde,params_dict)

        w = reshape(w,length(w),1)

        sg_B = 2 .*(w-reshape(l_s,length(l_s),1))

        @show(round.(sg_B,digits=2))

        sg_B =  (sg_B* reshape(f,1,length(f)))[1:params_dict["lookahead"],:]





        return sg_B

    end

    function initialize_forecaster(params_dict,warm_start)
        ndf = params_dict["n_diff_features"]
        pos_fc = params_dict["pos_fc"]
        la = params_dict["lookahead"]
        B_init = zeros(la,ndf*la)

        if warm_start
            for i in 1:la
                B_init[i,(i-1)*ndf+pos_fc] = -params_dict["eff_d"]
            end
        else
            #If no warm start, initialization has to be random because zeros leads to zero subgradient
            Random.seed!(73)
            B_init = -rand(la,ndf*la)
        end

        return B_init
    end

    function calc_subgrad_reg(B,λ,type="L1")

        if type == "L1"
            return λ.*sign.(B)
        else
            exit("Invalid type of regularization")
        end

    end



    features, labels_price, labels_schedule = first(training_loader)
    features, labels_price, labels_schedule = transpose(features), transpose(labels_price),transpose(labels_schedule)


    params_dict = dict_all["params_dict"]
    val_feat = dict_all["val_feat"]
    val_lab = dict_all["val_lab"]

    B = initialize_forecaster(params_dict,training_dict["warm_start"])
    @show(round.(B,digits=2))

    list_fc_lists = [[B]]
    list_val_profit = []


    e=1
    stale_epochs = 0

    while e < training_dict["SG_epochs"]

        batches = 0

        subgrad_spo = zeros(size(list_fc_lists[1][1]))

        for i in 1:size(features)[1]

            subgrad_spo += calc_subgrad_spo(features[i,:],labels_price[i,:],labels_schedule[i,:],B)
            batches+=1

        end

        subgrad_reg = calc_subgrad_reg(B,params_dict["reg"])

        subgrad_tot = subgrad_spo./batches + subgrad_reg

        if e == 1
            @show(round.(subgrad_tot,digits=2))
            @show(round.(features,digits=2))
        end



        γ = training_dict["lr"]/sqrt(e)
        @show(γ)

        B += γ * subgrad_tot

        # validation_profit = get_profit([B], params_dict, val_feat, val_lab)
        # println("Valdiation profit iteration: $(validation_profit)")
        # append!(list_val_profit,validation_profit)


        push!(list_fc_lists,[B])

        e+=1

    end

    return list_fc_lists

end




