function hp_tuning_spo_par(input_dict)
    #Overarching function of the training process, unpacking all the relevant data of the input dict, calling the training function, and performing post-training calculations, which are saved and returned in a dictionary
    
    #load data
    train_feat, train_lab, val_feat, val_lab, test_feat, test_lab = input_dict["list_tensors"]
    train_lab_training, val_lab_training, test_lab_training = train_lab, val_lab, test_lab

    params_dict = input_dict["params_dict"]
    training_dict = input_dict["training_dict"]
    model_type = input_dict["model_type"]

    params_dict["activation_function"] = training_dict["activation_function"] #For price / profit calculation

    outcome_dict = Dict(
    )

    input_dict_config = Dict(
        "model_type" => model_type,
        "val_test_feat" => [val_feat, test_feat],
        "val_test_lab_training" => [val_lab_training, test_lab_training],
        "dir" => input_dict["dir"] * "seq/",
        "params_dict" => params_dict,
        "training_dict" => training_dict,
        "val_feat" => val_feat,
        "val_lab_training" => val_lab_training,
        "test_feat" => test_feat,
        "test_lab_training" => test_lab_training
    )

    n_diff_features = params_dict["n_diff_features"]
    lookahead = params_dict["lookahead"]
    
    pos_fc = params_dict["pos_fc"]
    net_fc = zeros(Float64, lookahead, lookahead * n_diff_features)
    for i in 1:lookahead
        net_fc[i, (i - 1) * n_diff_features + pos_fc] = -params_dict["eff_d"]
    end

    batch_size=input_dict["batch_size"]
    reg = input_dict["reg"]
    lr = input_dict["lr"]
    restrict = input_dict["restrict"]
    pert = input_dict["pert"]
    hp_config = input_dict["hp_config"]
    ws = input_dict["warm_start"]

    #Define dataloader
    if model_type == "edRNN"
        if training_dict["type_train_labels"] == "price_schedule"
            train_Dataset = Flux.Data.DataLoader([train_feat[1], train_feat[2], train_lab_training[1], train_lab_training[2]], batchsize=batch_size, shuffle=true)
        else
            train_Dataset = Flux.Data.DataLoader([train_feat, train_lab_training], batchsize=batch_size, shuffle=true)
        end
    else
        if training_dict["type_train_labels"] == "price_schedule"
            train_Dataset = Flux.Data.DataLoader((transpose(train_feat), transpose(train_lab_training[1]), transpose(train_lab_training[2])), batchsize=batch_size, shuffle=true)
        else
            train_Dataset = Flux.Data.DataLoader([train_feat, train_lab_training], batchsize=batch_size, shuffle=true)
        end
    end


    input_dict_config["batch_size"] = batch_size
    input_dict_config["lr"] = lr
    input_dict_config["reg"] = reg
    input_dict_config["training_loader"] = train_Dataset
    input_dict_config["params_dict"]["restrict_forecaster_ts"] = restrict
    input_dict_config["params_dict"]["perturbation"] = pert
    input_dict_config["params_dict"]["reg"] = reg
    input_dict_config["training_dict"]["warm_start"] = ws
    input_dict_config["training_dict"]["SG_clip"] = calc_gradient_clip(training_dict,batch_size,size(train_feat)[1],lr)
    input_dict_config["val_feat"] = val_feat
    input_dict_config["val_lab"] = val_lab


    # Perfect foresight profit benchmark for calculating regret
    profit_PF_train = calc_profit_from_price_fc(train_lab[1],params_dict,train_lab[1])
    @show(profit_PF_train)
    profit_PF_val = calc_profit_from_price_fc(val_lab,params_dict,val_lab)
    @show(profit_PF_val)
    profit_PF_test = calc_profit_from_price_fc(test_lab,params_dict,test_lab)
    @show(profit_PF_test)


    tic=time()

    #Actual 'training' of the neural network via chosen procedure
    list_fc_lists, dict_evols, time_WS, time_opti, time_val = train_spo_new(input_dict_config)

    n_iters = length(list_fc_lists)
    iter_best = 0
    train_profit_best = -Inf
    val_profit_best = -Inf
    test_profit_best = -Inf
    best_net = nothing

    profit_evol_train = []
    profit_evol_val = []
    profit_evol_test = []

    regret_evol_train = []
    regret_evol_val = []
    regret_evol_test = []

    reg_evol = []



    # For all the intermediate outcomes accessed by the solver, calculate the downstream profit
    for (iter,net) in enumerate(list_fc_lists)

        train_profit = calc_profit_from_forecaster(net, params_dict, train_feat, train_lab[1])
        validation_profit = calc_profit_from_forecaster(net, params_dict, val_feat, val_lab)
        test_profit = calc_profit_from_forecaster(net, params_dict, test_feat, test_lab)

        println("Iter: $(iter), Train profits: $train_profit, Validation profits: $validation_profit, Test profits: $test_profit")

        if validation_profit > val_profit_best
            best_net = net
            iter_best = iter
            val_profit_best = validation_profit
            train_profit_best = train_profit
            test_profit_best=test_profit
            println("New best found at iter $iter_best; validation profit: $val_profit_best")
        end
        
        push!(profit_evol_train, train_profit)
        push!(profit_evol_val, validation_profit)
        push!(profit_evol_test, test_profit)
        push!(regret_evol_train,profit_PF_train-train_profit)
        push!(regret_evol_val,profit_PF_val-validation_profit)
        push!(regret_evol_test,profit_PF_test-test_profit)

        push!(reg_evol, get_reg_penalty(net,reg))


    end

    train_time = time()-tic

    println("Performance of best net: Train profits: $train_profit_best, Validation profits: $val_profit_best, Test profits: $test_profit_best")

    #Also calculate the final profit obtained by the initial forecaster (fc) and the last point accessed by the solver, i.e. the opimal solution found (opt)
    train_profit_fc = get_profit([net_fc], params_dict, train_feat, train_lab[1])
    validation_profit_fc = get_profit([net_fc], params_dict, val_feat, val_lab)
    test_profit_fc = get_profit([net_fc], params_dict, test_feat, test_lab)
    println("Train profits forecast: $train_profit_fc, Validation profits forecast: $validation_profit_fc, Test profits forecast: $test_profit_fc")

    train_profit_opt = get_profit(list_fc_lists[end], params_dict, train_feat, train_lab[1])
    validation_profit_opt = get_profit(list_fc_lists[end], params_dict, val_feat, val_lab)
    test_profit_opt = get_profit(list_fc_lists[end], params_dict, test_feat, test_lab)
    println("Train profits optimal: $train_profit_opt, Validation profits optimal: $validation_profit_opt, Test profits optimal: $test_profit_opt")
    

    #Save all the relevant outcomes and put them in a dictionary to be returned by the function
    dict_evols["profit_evol_train"] = profit_evol_train
    dict_evols["profit_evol_val"] = profit_evol_val
    dict_evols["profit_evol_test"] = profit_evol_test
    dict_evols["regret_evol_train"] = regret_evol_train
    dict_evols["regret_evol_val"] = regret_evol_val
    dict_evols["regret_evol_test"] = regret_evol_test
    dict_evols["reg_evol"] = reg_evol

    outcome_dict["dict_evols"] = dict_evols
    outcome_dict["list_fc_lists"] = list_fc_lists

    outcome_dict["net_best"] = best_net
    outcome_dict["net_opti"] = list_fc_lists[end]
    outcome_dict["a_config"] = hp_config
    outcome_dict["a_profit_train"] = train_profit_best
    outcome_dict["a_profit_val"] = val_profit_best
    outcome_dict["a_profit_test"] = test_profit_best
    outcome_dict["a_profit_train_fc"] = train_profit_fc
    outcome_dict["a_profit_val_fc"] = validation_profit_fc
    outcome_dict["a_profit_test_fc"] = test_profit_fc
    outcome_dict["a_profit_train_opt"] = train_profit_opt
    outcome_dict["a_profit_val_opt"] = validation_profit_opt
    outcome_dict["a_profit_test_opt"] = test_profit_opt
    outcome_dict["ab_profit_train_PF"] = profit_PF_train
    outcome_dict["ab_profit_val_PF"] = profit_PF_val
    outcome_dict["ab_profit_test_PF"] = profit_PF_test
    outcome_dict["b_time_all"] = train_time
    outcome_dict["b_time_WS"] = time_WS
    outcome_dict["b_time_opti"] = time_opti
    outcome_dict["b_time_val"] = time_val
    outcome_dict["b_train_time_without_WS"] = train_time - time_WS
    outcome_dict["b_n_iters"] = n_iters
    outcome_dict["b_iter_best"] = iter_best
    outcome_dict["hp_restrict_forecaster_ts"] = restrict
    outcome_dict["hp_lr"] = lr
    outcome_dict["hp_perturbation"] = pert
    outcome_dict["hp_reg"] = reg
    outcome_dict["hp_batch_size"] = batch_size



    return outcome_dict

end

function train_spo_new(dict)
    #Function calling either the subgradient training method, or the IP training method, returning the evolution of forecasters accessed by the respective solution procedures

    training_loader = dict["training_loader"]
    params_dict = dict["params_dict"]
    training_dict = dict["training_dict"]
    training_dict["lr"] = dict["lr"]
    train_type = training_dict["train_type"]

    if (train_type == "IP") || (train_type == "SL")

        obj_values = nothing
        list_mu = nothing
        train_time_WS = 0
        time_opti = 0
        all_list_fc_lists = Dict()
        all_lists_mu = Dict()

        for (i, data) in enumerate(training_loader)
            features, labels_price, labels_schedule = data
            features, labels_price, labels_schedule = transpose(features), transpose(labels_price),transpose(labels_schedule)


            labels_price_ext = zeros(Float64, size(labels_price, 1), size(labels_price, 2) * 3 + 1)
            for i in 1:size(labels_price, 1)
                labels_price_ext[i, :] = extend_price(labels_price[i, :],params_dict)
            end

            list_fc_lists, obj_values, list_mu, time_WS_batch, time_opti_batch = train_forecaster_spo(params_dict, training_dict, features, labels_price_ext, labels_schedule, dict,train_type, warm_start=training_dict["warm_start"])
            time_opti += time_opti_batch
            train_time_WS += time_WS_batch
            all_list_fc_lists[i] = list_fc_lists
            all_lists_mu[i] = list_mu
        end

        aggregator = Batch_aggregator(all_list_fc_lists,dict["val_feat"],dict["val_lab"],params_dict,"high","full_set","mu",all_lists_mu)
        tic = time()
        list_fc_lists = aggregate_batched_output(all_list_fc_lists,dict["val_feat"],dict["val_lab"],params_dict,"high","full_set","mu",all_lists_mu)
        time_val = time()-tic

        return list_fc_lists,Dict("obj_values" => obj_values, "list_mu" => list_mu),train_time_WS, time_opti, time_val
        #TODO: this currently just gives the results of the last batch --> adjust to see aggregated results
        
    
    elseif train_type == "SG"
        #list_fc_lists = subgradient_training(params_dict,training_dict,dict,training_loader)
        list_fc_lists = subgradient_training_flux(params_dict,training_dict,dict,training_loader)
        train_time_WS = 0

        return list_fc_lists, Dict(), train_time_WS

    else
        exit("Invalid train type")
    end 

end

function train_forecaster_spo(params_dict, training_dict, features, prices, optimal_schedules,dict_all,train_type;warm_start=false)
    
    function retrieve_callback_values(prob, B::JuMP.VariableRef)
        return callback_value(prob, B)
    end
    
    function retrieve_callback_values(prob, B::AbstractArray)
        return [retrieve_callback_values(prob, B[Tuple(i)...]) for i in CartesianIndices(B)]
    end
    
    function callback_intermediate_exp(
        alg_mod::Cint, iter_count::Cint, obj_value::Float64, inf_pr::Float64, inf_du::Float64, 
        mu::Float64, d_norm::Float64, regularization_size::Float64, 
        alpha_du::Float64, alpha_pr::Float64, ls_trials::Cint)
    
        #B_values_callback = retrieve_callback_values(prob, B)
        list_point = []
        for fc_item in list_fc
            callback_item = retrieve_callback_values(prob,fc_item)
            push!(list_point, callback_item)
        end


        push!(list_fc_lists, list_point)

        push!(obj_values, obj_value)
        push!(list_mu,mu)
        
        return iter_count < 500
    end

    function collect_points(cb_data, cb_where)
        if cb_where == GRB_CB_SIMPLEX
            # Extract the current solution, assuming it has more than 4 elements
            cur_sol = callback_value(cb_data, :SIMPLEX)
            @show(cur_sol)
            # Extract relevant elements for B and reshape to 2x2 matrix
            B_sol = cur_sol[1:4]
            push!(accessed_matrices, B_sol)
        end
    end

    function set_initial_values(list_variables,warm_start)
        #Function setting the initial values of the varaibles of the ERM to feasible points of a slightly modified ERM, where the output of the re-forecaster has to coincide with a pre-trained model
        pretrained_fc = training_dict["pretrained_fc"]


        if warm_start

            set_optimizer_attribute(prob, "warm_start_init_point", "yes")
            list_variables_feas = nothing

            if training_dict["train_mode"] == "linear"
                B=list_fc[1]
                ndf = params_dict["n_diff_features"]
                pos_fc = params_dict["pos_fc"]
                
                la = params_dict["lookahead"]
                pretrained_forecaster = zeros(la,ndf*la)
                for i in 1:la
                    pretrained_forecaster[i,(i-1)*ndf+pos_fc] = - params_dict["eff_d"]
                end

                prob_feas, list_fc_feas, list_variables_feas = spo_plus_erm_cvxpy_feasibility(params_dict=params_dict,training_dict = training_dict, examples=features, c=-prices, list_pretrained_fc=[pretrained_forecaster], train_type=train_type)
        
                for i in CartesianIndices(list_variables_feas[1])
                    set_start_value(list_variables_feas[1][Tuple(i)...], pretrained_forecaster[Tuple(i)...])
                end
                for i in CartesianIndices(list_variables_feas[2])
                    set_start_value(list_variables_feas[2][Tuple(i)...], abs(pretrained_forecaster[Tuple(i)...]))
                end



            elseif training_dict["train_mode"] == "nonlinear"

                prob_feas, list_fc_feas, list_variables_feas = spo_plus_erm_cvxpy_feasibility(params_dict=params_dict,training_dict = training_dict, examples=features, c=-prices, list_pretrained_fc=training_dict["pretrained_fc"],train_type=train_type)
                
                list_pfc = training_dict["pretrained_fc"]
                W1,b1,W2,b2 = list_pfc
                pretrained_model = training_dict["pretrained_model"]
                pretrained_prices = transpose(pretrained_model(transpose(features)))
                pretrained_z = transpose(pretrained_model[1](transpose(features)))

                pretrained_prices_ext = zeros(Float64, size(pretrained_prices, 1), size(pretrained_prices, 2) * 3 + 1)
                for i in 1:size(pretrained_prices, 1)
                    pretrained_prices_ext[i, :] = extend_price(pretrained_prices[i, :],params_dict)
                end                    

                for i in CartesianIndices(list_variables_feas[1])
                    set_start_value(list_variables_feas[1][Tuple(i)...], W1[Tuple(i)...])
                end
                for i in CartesianIndices(list_variables_feas[2])
                    set_start_value(list_variables_feas[2][Tuple(i)...], b1[Tuple(i)...])
                end
                for i in CartesianIndices(list_variables_feas[3])
                    set_start_value(list_variables_feas[3][Tuple(i)...], W2[Tuple(i)...])
                end
                for i in CartesianIndices(list_variables_feas[4])
                    set_start_value(list_variables_feas[4][Tuple(i)...], b2[Tuple(i)...])
                end
                for i in CartesianIndices(list_variables_feas[5])
                    set_start_value(list_variables_feas[5][Tuple(i)...], abs(W1[Tuple(i)...]))
                end
                for i in CartesianIndices(list_variables_feas[6])
                    set_start_value(list_variables_feas[6][Tuple(i)...], abs(W2[Tuple(i)...]))
                end
                for i in CartesianIndices(list_variables_feas[7])
                    set_start_value(list_variables_feas[7][Tuple(i)...], pretrained_z[Tuple(i)...])
                end
                for i in CartesianIndices(list_variables_feas[8])
                    set_start_value(list_variables_feas[8][Tuple(i)...], pretrained_prices_ext[Tuple(i)...])
                end
                
            end

            println("optimizing feasibility problem")
            optimize!(prob_feas)

            for (var,var_feas) in zip(list_variables, list_variables_feas)
                for i in CartesianIndices(var)
                    set_start_value(var[Tuple(i)...], value(var_feas[Tuple(i)...]))
                end
            end



        else 

            nothing

        end


    end

    function solve_with_custom_mu(model::Model, initial_mu, update_mu_func, dict_all, tol=1e-3, max_iters=50,patience=5,max_iters_single_opti=100, lim_mu = -8)
        
        #Retrieve values from overall dict
        params_dict = dict_all["params_dict"]
        val_feat = dict_all["val_feat"]
        val_lab = dict_all["val_lab"]
        
        #Initialize arrays to keep track of progress
        list_fc_lists = []
        list_vars = []
        list_obj_values = []
        list_val_profit = []
        list_mu = [initial_mu]
        
        # Set IPopt options
        
        iter = 0
        iter_suboptimal = 0
        curr_mu = initial_mu
        previous_obj_value = Inf
        while (iter < max_iters) & (log10(curr_mu) > lim_mu)
            @show(curr_mu)
            # Set the current mu value
            set_optimizer_attribute(model, "warm_start_init_point", "yes")
            set_optimizer_attribute(model, "mu_target", curr_mu)
            set_optimizer_attribute(model, "mu_init", curr_mu)
            set_optimizer_attribute(model, "max_iter", max_iters_single_opti)

   
            # Solve the problem with the current mu
            optimize!(model)
    
            # Check convergence
            current_obj_value = value(objective_function(model))
            if (abs(current_obj_value - previous_obj_value) < tol) && (iter_suboptimal>=patience)
               println("Previous objective value: $(previous_obj_value); Current objective value: $(current_obj_value)")
               break
            elseif (abs(current_obj_value - previous_obj_value) >= tol)
                iter_subotpimal = 0
            else
                iter_suboptimal += 1
            end
    

            fc_list_point = []
            for fc_item in list_fc
                fc_item = value.(fc_item)
                push!(fc_list_point, fc_item)
            end
            push!(list_fc_lists,fc_list_point)
            push!(list_obj_values, value(objective_function(model)))
            push!(list_vars,retrieve_optimal_values(list_variables))

   
            # Transfer primal and dual solutions as starting points for next solve
            # Store the primal and dual solutions in local variables
            primal_values = Dict(v => value(v) for v in all_variables(model))
            constraint_types = list_of_constraint_types(model)
            dual_values = Dict{Any, Any}()
            for (F, S) in constraint_types
               for c in all_constraints(model, F, S)
                   dual_values[c] = dual(c)
               end
            end
           
           # Transfer primal and dual solutions as starting points for next solve
            for v in all_variables(model)
               set_start_value(v, primal_values[v])
            end
            for (c, dual_val) in dual_values
               if isa(dual_val, Number) # Scalar constraint
                   set_dual_start_value(c, dual_val)
               else # For vector-valued constraints
                   for (i, d_val) in enumerate(dual_val)
                       set_dual_start_value(c, i, d_val)
                   end
               end
            end

            #calculate validation profit
            validation_profit = get_profit(fc_list_point, params_dict, val_feat, val_lab)
            println("Valdiation profit iteration: $(validation_profit)")
            append!(list_val_profit,validation_profit)


            # Increase the iteration count
            iter += 1
            previous_obj_value = current_obj_value   
            curr_mu,divisor = update_mu_func(curr_mu,list_val_profit)
            append!(list_mu,curr_mu)

            if validation_profit == maximum(list_val_profit)
                iter_suboptimal = 0
            else
                iter_suboptimal +=1
            end

   
   
        end
    
        return list_fc_lists,list_obj_values,list_mu,list_vars
   
    end

    function retrieve_optimal_values(list_variables)
        list_optimized_variables = []
        for var in list_variables
            push!(list_optimized_variables,[value(var[Tuple(i)...]) for i in CartesianIndices(var)])
        end
        return list_optimized_variables
    end


    list_fc_lists = []
    obj_values = []
    list_mu = []

    prob, list_fc, list_variables = spo_plus_erm_cvxpy_new(params_dict=params_dict,training_dict = training_dict, examples=features, c=-prices, optimal_schedules=optimal_schedules, train_type=train_type)

    tic = time()
    set_initial_values(list_variables,warm_start)
    time_WS = time()-tic

    if train_type == "IP"
        set_optimizer_attribute(prob,"nlp_scaling_method", "none")    
        #set_optimizer_attribute(prob, "hessian_approximation", "limited-memory")
    end

    if training_dict["mu_update"] == "auto"

        # Set the callback function

        if train_type == "IP"
            MOI.set(prob, Ipopt.CallbackFunction(), callback_intermediate_exp)
        elseif train_type == "SL"
            set_optimizer_attribute(prob, "SimplexCallback", collect_points)
        end
        println("***** STARTING TRAINING *****")
        tic=time()
        memory = @allocated optimize!(prob)
        time_opti = time()-tic
        println("Allocated memory: $(memory)")

    elseif training_dict["mu_update"][1:6] == "manual"

        mu_init = 10.0
        update_mu_func(curr_mu) = curr_mu / 2
        function update_mu_func_dyn(mu,list_val_profit)
            divisor = 1.5
            if training_dict["mu_update"][end:end] == "s"
                dec = 1.0
            elseif training_dict["mu_update"][end:end] == "d"
                dec = 0.9
            else
                exit("Unsupported mu update setting")
                exit("$(training_dict["mu_update"]) is an invalid mu update setting")
            end
            print("dec = $(dec)")

            
            if length(list_val_profit)>1
                if list_val_profit[end] > list_val_profit[end-1]
                    divisor *= dec
                end
                if list_val_profit[end] > maximum(list_val_profit[1:end-1])
                    divisor *= dec
                end
            end
            return mu/divisor, divisor
        end
        tic=time()
        list_fc_lists,obj_values,list_mu,list_vars = solve_with_custom_mu(prob, mu_init, update_mu_func_dyn,dict_all)
        time_opti = time()-tic

    end    
    
    return list_fc_lists, obj_values, list_mu, time_WS, time_opti
end

function spo_plus_erm_cvxpy_new(;params_dict, training_dict, examples, c, optimal_schedules,train_type)
    #Function returning optimization program defining the ERM, together with a list of the variables that define the forecaster, and a list of all variables
   
    A, b = get_full_matrix_problem_DA(params_dict)
    A = sparse(A)

    lookahead = params_dict["lookahead"]
    eff = params_dict["eff_d"]
    reg = params_dict["reg"]
    perturbation = params_dict["perturbation"]
    n_diff_features = params_dict["n_diff_features"]
    pos_fc = params_dict["pos_fc"]
    restrict_forecaster_ts = params_dict["restrict_forecaster_ts"]


    n_constraints = size(A, 1)
    prob_dim = size(A, 2)
    n_examples = size(examples, 1)
    n_features = size(examples, 2)

    Random.seed!(73)

    model=nothing
    if train_type == "IP"
        model = Model(Ipopt.Optimizer)
    elseif train_type == "SL"
        model = Model.(Gurobi.Optimizer)
    else
        error("Invalid train type")
    end

    @variable(model, p[1:n_examples, 1:n_constraints] >= 0)
    @variable(model, c_hat[1:n_examples,1:prob_dim])

    if training_dict["train_mode"] == "linear"

        @variable(model, B[1:lookahead, 1:n_features])
        @variable(model, B_abs[1:lookahead, 1:n_features] >= 0)
        

        @constraint(model, B_abs .>= B)
        @constraint(model, B_abs .>= -B)

        @constraint(model,B_abs .<= 10)


        @constraint(model,[i=1:n_examples,j=1:lookahead], c_hat[i,j] == sum(examples[i,k]*B[j,k] for k in 1:n_features))


        if restrict_forecaster_ts
            for i in 1:lookahead
                for j in 1:n_features
                    if j < n_diff_features * (i - 1) + 1
                        @constraint(model, B[i, j] == 0)
                    elseif j >= n_diff_features * i + 1
                        @constraint(model, B[i, j] == 0)
                    end
                end
            end
        end

        @objective(
            model,
            Min,
            -sum(p * b)+ 2 * tr(optimal_schedules * transpose(c_hat)) + reg * sum(B_abs)
        )

        list_forecaster = [B]
        list_variables = [B,B_abs,c_hat,p]

    

    elseif training_dict["train_mode"] == "nonlinear"

        nhu_1 = lookahead
        hu_per_la = Int(nhu_1/lookahead)

        @variable(model, W1[1:nhu_1, 1:n_features])
        @variable(model,b1[1:nhu_1])
        @variable(model,W2[1:lookahead,1:nhu_1])
        @variable(model,b2[1:lookahead])
        @variable(model,W1_abs[1:nhu_1, 1:n_features])
        @variable(model,W2_abs[1:lookahead,1:nhu_1])

        @variable(model,z1[1:n_examples,1:nhu_1])

        @constraint(model, W1_abs .>= W1)
        @constraint(model, W1_abs .>= -W1)

        @constraint(model, W2_abs .>= W2)
        @constraint(model, W2_abs .>= -W2)


        if training_dict["activation_function"] == "relu"
            @NLconstraint(model,[i=1:n_examples,j=1:nhu_1], z1[i,j] == max(sum(examples[i,k]*W1[j,k] for k in 1:n_features)  + b1[j],0))
        elseif training_dict["activation_function"] == "softplus"
            @NLconstraint(model,[i=1:n_examples,j=1:nhu_1], z1[i,j] == log(1 + exp( sum(examples[i,k]*W1[j,k] for k in 1:n_features)  + b1[j] ) ) )
        elseif training_dict["activation_function"] == "sigmoid"
            println("Correct constraint prob")
            @NLconstraint(model,[i=1:n_examples,j=1:nhu_1], z1[i,j] == 1/(1 + exp( -sum(examples[i,k]*W1[j,k] for k in 1:n_features)  - b1[j] ) ) )
        end


        @constraint(model,W1_abs .<= 5)
        @constraint(model,W2_abs .<= 5)
        @constraint(model,b1 .>= -5)
        @constraint(model,b1 .<= 5)
        @constraint(model,b2 .>= -5)
        @constraint(model,b2 .<= 5)


        @constraint(model,[i=1:n_examples,j=1:lookahead], c_hat[i,j] == sum(z1[i,k]*W2[j,k] for k in 1:nhu_1) + b2[j])


        if restrict_forecaster_ts
            for i in 1:nhu_1
                for j in 1:n_features
                    if j < n_diff_features * (i - 1) + 1
                        @constraint(model, W1[i, j] == 0)
                    elseif j >= n_diff_features * i + 1
                        @constraint(model, W1[i, j] == 0)
                    end
                end
                for la in 1:lookahead
                    if i < hu_per_la * (la - 1) + 1
                        @constraint(model, W2[la,i] == 0)
                    elseif i >= hu_per_la * la + 1
                        @constraint(model, W2[la,i] == 0)
                    end
                end
            end
        end

        @objective(
            model,
            Min,
            -sum(p * b)+ 2 * tr(optimal_schedules * transpose(c_hat))  + reg * (sum(W1_abs)+sum(W2_abs))
        )


        list_forecaster = [W1,b1,W2,b2]
        list_variables = [W1,b1,W2,b2,W1_abs,W2_abs,z1,c_hat,p]

    end


    @constraint(model, [i=1:n_examples, j=1:prob_dim], sum(p[i, k] * A[k,j] for k in 1:n_constraints) == 2* c_hat[i,j] - c[i, j])
    @constraint(model, c_hat[:,2 * lookahead + 1:end] .== 0)
    @constraint(model, c_hat[:,1:lookahead] .== -c_hat[:,lookahead+1:2*lookahead]*(eff^2))


    if perturbation > 0
        forecast_c = zeros(n_examples, lookahead)
        for ex in 1:n_examples
            for la in 1:lookahead
                forecast_c[ex, la] = -examples[ex, pos_fc + (la - 1) * n_diff_features]
            end
        end
        @constraint(model, c_hat[:,1:lookahead] .>= forecast_c*eff - perturbation * abs.(forecast_c*eff))
        @constraint(model, c_hat[:,1:lookahead] .<= forecast_c*eff + perturbation * abs.(forecast_c*eff))
    end

    return model,list_forecaster,list_variables

end

function spo_plus_erm_cvxpy_feasibility(;params_dict, training_dict, examples, c, list_pretrained_fc, train_type)
    # Defining a feasibility optimization program to warm start the ERM
   
    A, b = get_full_matrix_problem_DA(params_dict)
    A = sparse(A)

    lookahead = params_dict["lookahead"]
    eff = params_dict["eff_d"]
    reg = params_dict["reg"]
    perturbation = params_dict["perturbation"]
    n_diff_features = params_dict["n_diff_features"]
    pos_fc = params_dict["pos_fc"]
    restrict_forecaster_ts = params_dict["restrict_forecaster_ts"]


    n_constraints = size(A, 1)
    prob_dim = size(A, 2)
    n_examples = size(examples, 1)
    n_features = size(examples, 2)

    model=nothing
    if train_type == "IP"
        model = Model(Ipopt.Optimizer)
    elseif train_type == "SL"
        model = Model.(Gurobi.Optimizer)
    else
        error("Invalid train type")
    end

    @variable(model, p[1:n_examples, 1:n_constraints] >= 0)
    @variable(model, c_hat[1:n_examples,1:prob_dim])

    if training_dict["train_mode"] == "linear"

        @variable(model, B[1:lookahead, 1:n_features])
        @variable(model, B_abs[1:lookahead, 1:n_features] >= 0)
        
        @constraint(model, B .== list_pretrained_fc[1])
        @constraint(model, [i=1:size(B_abs, 1), j=1:size(B_abs, 2)], B_abs[i,j] == abs(list_pretrained_fc[1][i,j]))

        @constraint(model,[i=1:n_examples,j=1:lookahead], c_hat[i,j] == sum(examples[i,k]*B[j,k] for k in 1:n_features))

        list_forecaster = [B]
        list_variables = [B,B_abs,c_hat,p]

    

    elseif training_dict["train_mode"] == "nonlinear"

        nhu_1 = lookahead
        #hu_per_la = Int(nhu_1/lookahead)

        @variable(model, W1[1:nhu_1, 1:n_features])
        @variable(model,b1[1:nhu_1])
        @variable(model,W2[1:lookahead,1:nhu_1])
        @variable(model,b2[1:lookahead])
        @variable(model,W1_abs[1:nhu_1, 1:n_features])
        @variable(model,W2_abs[1:lookahead,1:nhu_1])

        @variable(model,z1[1:n_examples,1:nhu_1])

        @constraint(model, [i=1:size(W1_abs, 1), j=1:size(W1_abs, 2)], W1_abs[i,j] == abs(list_pretrained_fc[1][i,j]))
        @constraint(model, [i=1:size(W2_abs, 1), j=1:size(W2_abs, 2)], W2_abs[i,j] == abs(list_pretrained_fc[3][i,j]))

        @constraint(model, W1 .== list_pretrained_fc[1])
        @constraint(model, b1 .== list_pretrained_fc[2])
        @constraint(model, W2 .== list_pretrained_fc[3])
        @constraint(model, b2 .== list_pretrained_fc[4])


        if training_dict["activation_function"] == "relu"
            @NLconstraint(model,[i=1:n_examples,j=1:nhu_1], z1[i,j] == max(sum(examples[i,k]*W1[j,k] for k in 1:n_features)  + b1[j],0))
        elseif training_dict["activation_function"] == "softplus"
            @NLconstraint(model,[i=1:n_examples,j=1:nhu_1], z1[i,j] == log(1 + exp( sum(examples[i,k]*W1[j,k] for k in 1:n_features)  + b1[j] ) ) )
        elseif training_dict["activation_function"] == "sigmoid"
            @NLconstraint(model,[i=1:n_examples,j=1:nhu_1], z1[i,j] == 1/(1 + exp( -sum(examples[i,k]*W1[j,k] for k in 1:n_features)  - b1[j] ) ) )
        end

        @constraint(model,[i=1:n_examples,j=1:lookahead], c_hat[i,j] == sum(z1[i,k]*W2[j,k] for k in 1:nhu_1) + b2[j])
    
        list_forecaster = [W1,b1,W2,b2]
        list_variables = [W1,b1,W2,b2,W1_abs,W2_abs,z1,c_hat,p]


    end


    @constraint(model, [i=1:n_examples, j=1:prob_dim], sum(p[i, k] * A[k,j] for k in 1:n_constraints) == 2* c_hat[i,j] - c[i, j])
    @constraint(model, c_hat[:,2 * lookahead + 1:end] .== 0)
    @constraint(model, c_hat[:,1:lookahead] .== -c_hat[:,lookahead+1:2*lookahead]*(eff^2))


    return model, list_forecaster, list_variables

end

function aggregate_batched_output(dict_list_fc_lists,feat,lab,params_dict,mode,agg_type,eval_type,dict_list_mu)
    """
    Function that aggregates a list of outputs of training outputs of individual batches into a single forecaster based on their performance on a specified dataset
    # Args
    - dict_list_fc_lists: a dict with the evolution of forecaster list how they evolved over the training for a specific batch, which is assigned an integer key
    - feat: features (typically of a validation set) that serve as inputs to the forecaster
    - lab: labels of that same set
    - params_dict: dictionary containing the required information for calculating performance, such as optimization setting
    - mode: determines whether the performance should be as low or high as possible
    - agg_type: String specifying the method of aggregation. 
    - eval_type: String specifying the evaluation: "mu" or "newton" 
    - dict_list_mu: Dictionary with same keys as dict_list_fc_lists with the evolution of log-barrier weights during the training of the respective batch

    # Output
    - fc_list: a single list containing the information of a single forecaster being the aggregate of the input 
    """

    function get_mul(mode)
        if mode == "high" #The best performer has the highest value
            return 1
        elseif mode == "low" #The best performer has the lowest value
            return -1
        end
    end

    function check_lower_mu(list_mu_batch,i)
        """
        Function checking if the current index i represents a local 'solution' to the barrier problem, i.e. whether or not the next mu is strictly smaller
        """
        n = length(list_mu_batch)
        if i<n
            return list_mu_batch[i+1]<list_mu_batch[i]
        else
            return true
        end
    end

    function get_best_performance_batch(list_fc_lists_batch,list_mu_batch,feat,lab,mul)
        perfo_best_batch = -Inf*mul
        index_best_batch = 1
        fc_best_batch = list_fc_lists_batch[1]

        for (i,fc) in enumerate(list_fc_lists_batch)
            println("Iteration $(i)")
            perfo = -Inf*mul
            if eval_type == "mu"
                if check_lower_mu(list_mu_batch,i)
                    println("Evluating validation performance")
                    perfo = calc_profit_from_forecaster(fc, params_dict, feat, lab)
                end
            elseif eval_type == "newton"
                perfo = calc_profit_from_forecaster(fc, params_dict, feat, lab)
            else
                exit("$(eval_type) is an unsupported evaluation type for calculating validation performance")
            end
            if mul*perfo > mul*perfo_best_batch
                println("***** New best found: updating validation performance from $(perfo_best_batch) to $(perfo) *****")
                perfo_best_batch = perfo
                index_best_batch = i
                fc_best_batch = fc
            end
        end

        return index_best_batch,fc_best_batch,perfo_best_batch
    end

    if agg_type == "full_set"

        mul = get_mul(mode)
        perfo_best = -Inf*mul
        index_best = 1
        fc_best = dict_list_fc_lists[1][1]

        for (i,batch_outcome) in pairs(dict_list_fc_lists)
            println("")
            println("Starting validation calculation batch $(i)")
            println("")
            mu_batch = dict_list_mu[i]
            index,fc,perfo = get_best_performance_batch(batch_outcome,mu_batch,feat,lab,mul)
            if mul*perfo >= mul*perfo_best
                println("Updating current best value $(perfo_best) to $(perfo)")
                perfo_best = perfo
                index_best = (i,index)
                fc_best = fc
            end
        end      
        return [fc_best]
    else
        sys.exit("Unsupported batch aggregation type.")
    end
end

mutable struct Batch_aggregator
    dict_list_fc_lists::Dict
    feat::Array
    lab::Array
    params_dict::Dict
    mode::String
    agg_type::String
    eval_type::String
    dict_list_mu::Dict

    function Batch_aggregator(dict_list_fc_lists::Dict,feat::Array,lab::Array,params_dict::Dict,mode::String,agg_type::String,eval_type::String,dict_list_mu::Dict)
        new(dict_list_fc_lists,feat,lab,params_dict,mode,agg_type,eval_type,dict_list_mu)
    end

end


function get_params_dict(obj::Batch_aggregator)
    return obj.params_dict
end