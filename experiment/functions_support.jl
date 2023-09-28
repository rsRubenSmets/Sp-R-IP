##### FUNCTIONS RELATING TO THE DOWNSTREAM OPTIMIZATION PROGRAM, PRICE AND PROFIT CALCULATION #####

function opti_da_cvxpy(params_dict, prices, solver)
    A, b = get_full_matrix_problem_DA(params_dict)

    lookahead = params_dict["lookahead"]
    model = nothing
    if solver == "IP"
        model = Model(Ipopt.Optimizer)
        set_silent(model)
    elseif solver == "Gurobi"
        model = Model(optimizer_with_attributes(Gurobi.Optimizer, "OutputFlag" => false))
    end
    @variable(model, x[1:lookahead * 3 + 1])
    @constraint(model,A*x .>= b)
    @objective(model,Max,sum(prices[k]*x[k] for k in 1:lookahead*3+1))

    return model,x
end

function get_full_matrix_problem_DA(params_dict)
    #Current definition assumes cyclic boundary conditions
    D_out = params_dict["lookahead"]
    D_in = D_out
    eff_c = params_dict["eff_c"]
    eff_d = params_dict["eff_d"]
    soc_0 = params_dict["soc_0"]
    soc_max = params_dict["max_soc"]
    soc_min = params_dict["min_soc"]
    max_charge = params_dict["max_charge"]


    A = zeros(7 * D_out + 4 + 2, 3 * D_out + 1)
    b = zeros(7 * D_out + 4 + 2)

    start_d = 0
    start_c = D_out 
    start_soc = 2 * D_out 

    # Positivity constraints
    for t in 1:D_out
        A[t + start_d, t + start_d] = 1
        A[t + start_c, t + start_c] = 1
        A[t + start_soc, t + start_soc] = 1
    end
    A[D_out + start_soc + 1, D_out + start_soc] = 1

    # Constrain max power
    start_constrain_max_power = 3 * D_out +1
    for t in 1:D_out
        A[t + start_constrain_max_power, t + start_d] = -1
        A[t + start_constrain_max_power, t + start_c] = -1
        b[t + start_constrain_max_power] = -max_charge
    end

    # Constrain max soc
    start_constrain_soc = 4 * D_out +1
    for t in 1:D_out
        A[t + start_constrain_soc, t + start_soc] = -1
        b[t + start_constrain_soc] = -soc_max
    end
    A[D_out + start_constrain_soc + 1, D_out + start_soc] = -1
    b[D_out + start_constrain_soc + 1] = -soc_max

    # SoC update
    start_soc_update_pos = 5 * D_out + 2
    start_soc_update_neg = 6 * D_out + 3
    for t in 1:D_out
        if t == 1
            A[t + start_soc_update_pos, t + start_soc] = 1
            b[t + start_soc_update_pos] = soc_0

            A[t + start_soc_update_neg, t + start_soc] = -1
            b[t + start_soc_update_neg] = -soc_0
        else
            A[t + start_soc_update_pos, t + start_soc] = 1
            A[t + start_soc_update_pos, t - 1 + start_soc] = -1
            A[t + start_soc_update_pos, t - 1 + start_d] = 1
            A[t + start_soc_update_pos, t - 1 + start_c] = -1

            A[t + start_soc_update_neg, t + start_soc] = -1
            A[t + start_soc_update_neg, t - 1 + start_soc] = 1
            A[t + start_soc_update_neg, t - 1 + start_d] = -1
            A[t + start_soc_update_neg, t - 1 + start_c] = 1
        end

        #Last update

        A[D_out + start_soc_update_pos + 1, D_out + start_soc + 1] = 1
        A[D_out + start_soc_update_pos + 1, D_out + start_soc] = -1
        A[D_out + start_soc_update_pos + 1, D_out + start_d] = 1
        A[D_out + start_soc_update_pos + 1, D_out + start_c] = -1

        A[D_out + start_soc_update_neg + 1, D_out + start_soc+1] = -1
        A[D_out + start_soc_update_neg + 1, D_out + start_soc] = 1
        A[D_out + start_soc_update_neg + 1, D_out + start_d] = -1
        A[D_out + start_soc_update_neg + 1, D_out + start_c] = 1
    end

    A[D_out + start_soc_update_neg + 2, D_out + start_soc + 1] = 1
    b[D_out + start_soc_update_neg + 2] = soc_0

    A[D_out + start_soc_update_neg + 3, D_out + start_soc+1] = -1
    b[D_out + start_soc_update_neg + 3] = -soc_0

    return A, b
end

function extend_price(price,params_dict,eff=true)
    lookahead = size(price, 1)
    eff_d = params_dict["eff_d"]
    eff_c = params_dict["eff_c"]

    extended_price = zeros(Float64, 3 * lookahead + 1)
    for la in 1:lookahead
        if eff
            extended_price[la] = price[la]*eff_d
            extended_price[lookahead + la] = -price[la]/eff_c
        else
            extended_price[la] = price[la]
            extended_price[lookahead + la] = -price[la]
        end
    end

    return extended_price
end

function extend_price_full(price_full,params_dict)
    n_examples = size(price_full,1)
    extended_price_full = zeros(n_examples,size(price_full,2)*3+1)

    for ex in 1:n_examples
        extended_price_full[ex,:] = extend_price(price_full[ex,:],params_dict)
    end

    return extended_price_full
end

function calc_optimal_schedule(prices, OP_params_dict, solver="IP", extended_price=false)
    n_examples = size(prices, 1)
    lookahead = OP_params_dict["lookahead"]
    optimal_schedule = zeros(n_examples, 3 * lookahead + 1)

    for ex in 1:n_examples
        if extended_price
            price = prices[ex, :]
        else
            price = extend_price(prices[ex, :],OP_params_dict)
        end

        model,x = opti_da_cvxpy(OP_params_dict,price,solver)
        set_silent(model)
        optimize!(model)

        optimal_schedule[ex, :] = value.(x)
    end

    return optimal_schedule
end

function calc_price_from_net(net_list,input_features,act_function)

    #Assumes that net_list is structured as [W1,b1,W2,b2,...,Wn,bn]
    #Weights: net_list[(l-1)*2+1 for l in 1:n_lays]
    #biases: net_list[(l-1)*2+2 for l in 1:n_lays]


    if length(net_list) == 1
        int_price = input_features * transpose(net_list[1])

    else
        n_lays = Int(length(net_list)/2)
        int_price = input_features
        for l in 1:n_lays
            int_price = transpose(broadcast(+,net_list[(l-1)*2 + 2], net_list[(l-1)*2+1]*transpose(int_price) ))
            if l!=n_lays
                if act_function == "softplus"
                    int_price = log.(1 .+ exp.(int_price))
                elseif act_function == "quadr"
                    int_price = int_price.^2
                elseif act_function == "relu"
                    int_price = max.(int_price,0)
                elseif act_function == "simgoid"
                    int_price = 1 ./(1 .+exp.(-int_price))
                end
            end
        end
    end

    return -int_price


end

function get_profit(net_list, input_parameters, input_features, input_labels)
    optimal_schedule = get_optimal_schedule(net_list, input_parameters, input_features)

    price_act_ext = zeros(size(input_labels)[1], size(input_labels)[2] * 3 + 1)
    for i in 1:size(input_labels, 1)
        price_act_ext[i, :] = extend_price(input_labels[i, :],input_parameters)
    end

    profit_per_timestep = price_act_ext .* optimal_schedule

    return sum(profit_per_timestep)
end

function get_reg_penalty(net_list,lambda,mode="abs")
    #Assuming net_list is of the form [W1,b1,W2,b2,...,Wn,bn]

    len_fc_list = length(net_list)

    reg = 0
    i = 1
    while 2*(i-1) + 1 <= len_fc_list
        if mode == "abs"
            reg += sum(abs.(net_list[Int(2*(i-1) + 1)]))
        end
        i+=1
    end

    return lambda*reg
end

function get_optimal_schedule(net_list, input_parameters, input_features)
    lookahead = input_parameters["lookahead"]


    price_fc = calc_price_from_net(net_list,input_features,input_parameters["activation_function"])
    optimal_schedule = calc_optimal_schedule(price_fc[:, 1:lookahead]/input_parameters["eff_d"], input_parameters)

    return optimal_schedule
end

function calculate_spo_plus(list_net,params_dict,feat,prices,opt_scheds)

    lambda_hat = calc_price_from_net(list_net,feat,params_dict["activation_function"])
    lambda_hat_extended = extend_price_full(lambda_hat,params_dict)
    c_hat = -lambda_hat_extended

    lambda_extended = extend_price_full(prices,params_dict)
    c = -lambda_extended

    input_opti = c-2*c_hat

    optimized_decisions = calc_optimal_schedule(input_opti,params_dict,"IP", true)

    spo_plus = tr(input_opti * transpose(optimized_decisions) + 2*c_hat*transpose(opt_scheds))

    return spo_plus



end

function calc_gradient_clip(training_dict,batch_size,n_examples,lr)

    epochs = training_dict["SG_epochs"]
    clip_base = training_dict["SG_clip_base"]

    n_grad_updates = epochs * n_examples / batch_size
    clip = clip_base / (n_grad_updates/100 * lr)

    return clip

end

function calc_profit_from_forecaster(net_list,input_parameters,input_features,input_labels)
    #Uses features and forecaster in list form to calculate predicted prices, turns those prices in optimized schedule, and calculates ex-post profits

    la = input_parameters["lookahead"]
    price_fc = calc_price_from_net(net_list,input_features,input_parameters["activation_function"])

    profit = calc_profit_from_price_fc(price_fc,input_parameters,input_labels)

    return profit
end

function calc_profit_from_price_fc(price_fc,input_parameters,input_labels,extended_price=false)
    #Input price fc is not extended, and is not adjusted for efficiency

    optimized_sched = calc_optimal_schedule(price_fc,input_parameters,"IP",extended_price)

    price_act_ext = zeros(size(input_labels)[1], size(input_labels)[2] * 3 + 1)
    for i in 1:size(input_labels, 1)
        price_act_ext[i, :] = extend_price(input_labels[i, :],input_parameters)
    end

    profit = sum(optimized_sched.*price_act_ext)

    return profit

end




####### FUNCTIONS FOR DATA PRE-PROCESSING #######

function get_indices(last_ex_test, days_train, train_share, tot_n_examples,val_split_mode)
    idx_test = [tot_n_examples-i for i in 0:last_ex_test-1]

    start = tot_n_examples - last_ex_test - days_train

    idx_train = []
    idx_val = []

    if val_split_mode == "separate"
        idx_train = [i for i in 1:Int(round(days_train*train_share))]
        idx_val = [i for i in Int(round(days_train*train_share))+1:days_train]

    elseif val_split_mode == "alternating"
        idx_train = [1]
        idx_val = [2]

        for i in 1:days_train-2
            share = length(idx_train) / (length(idx_train) + length(idx_val))

            if share < train_share
                push!(idx_train, i + 2)
            else
                push!(idx_val, i + 2)
            end
        end

    elseif val_split_mode =="alt_test"

        idx_train = [i for i in 1:Int(round(days_train*train_share))]
        idx_val = [Int(round(days_train*train_share)) + 2*i for i in 1:Int(round(last_ex_test/2)-1)]
        idx_test= [Int(round(days_train*train_share)) + 2*(i-1)+1 for i in 1:Int(round(last_ex_test/2)-1)]

        idx_test = [idx + start for idx in idx_test]


    end



    idx_train = [idx + start for idx in idx_train]
    idx_val = [idx + start for idx in idx_val]

    return idx_train, idx_val, idx_test
end

function retrieve_optimal_schedule_train_test(train_prices, validation_prices, OP_params_dict)

    optimized_train_sched = calc_optimal_schedule(train_prices, OP_params_dict)
    optimized_val_sched = calc_optimal_schedule(validation_prices, OP_params_dict)

    return optimized_train_sched, optimized_val_sched
end

function get_data_pandas(loc_data)
    data_forecast = CSV.read(loc_data * "forecast_df.csv", DataFrame)
    data = CSV.read(loc_data * "X_df_ds.csv", DataFrame)

    data[!, "dt"] = [ds[1:19] for ds in data[!,"ds"]]
    data[!, "Datetime"] = DateTime.(data[!, "dt"],dateformat"yyyy-mm-dd HH:MM:SS")
    data[!, "date"] = Date.(data[!, "Datetime"])

    data_forecast[!, "dt"] = [ds[1:19] for ds in data_forecast[!,"ds"]]
    data_forecast[!, "Datetime"] = DateTime.(data_forecast[!, "dt"],dateformat"yyyy-mm-dd HH:MM:SS")
    data_forecast[!, "date"] = Date.(data_forecast[!, "Datetime"])

    dates_rem = [
        Date(2019, 3, 18), Date(2019, 3, 31), Date(2020, 3, 29),
        Date(2021, 3, 28), Date(2022, 3, 27), Date(2023, 3, 26),
        Date(2019, 10, 27), Date(2020, 10, 25), Date(2021, 10, 31),
        Date(2022, 10, 30)
    ]

    filter_indices = .!([x in dates_rem for x in data[!, "date"]])
    filter_indices_fc = .!([x in dates_rem for x in data_forecast[!, "date"]])

    data = data[filter_indices, :]
    data_forecast = data_forecast[filter_indices_fc, :]

    data_all = innerjoin(data_forecast, data, on = "Datetime",makeunique=true)

    return data_all
end


function scale_df(df, mode; base=false)
    # When using a base, this column is not scaled. Other columns are rescaled, integrating the standard deviation of the base column to have similar spreads


    df_scaled = deepcopy(df)

    if typeof(base) != String

        for col in names(df)
            if eltype(df[!, col]) <: Number
                df_scaled[!, col] = scale_column(df[!, col], mode)
            end
        end
    else
        stdev_base = std(df[:,base])
        for col in names(df)
            if col != base && eltype(df[!, col]) <: Number
                df_scaled[!, col] = scale_column(df[!, col], mode)*stdev_base
            end
        end
    end

    return df_scaled
end

function scale_df_new(df,mode,cols_no_centering)
    df_scaled = deepcopy(df)


    for col in names(df)
        if eltype(df[!, col]) <: Number
            if col in cols_no_centering
                df_scaled[!,col] = scale_column(df[!, col], "stand_no_centering")
            else
                df_scaled[!, col] = scale_column(df[!, col], mode)
            end
        end
    end

    stdev_yhat = nothing
    if "y_hat" in names(df)
        stdev_yhat = std(df[!,"y_hat"])
    end
    return df_scaled, stdev_yhat
end


function scale_column(column, mode)
    if mode == "stand"
        scaled_column = (column .- mean(column)) ./ std(column)
    elseif mode == "stand_no_centering"
        scaled_column = column ./ std(column)
    elseif mode == "norm"
        scaled_column = (column .- minimum(column)) ./ (maximum(column) - minimum(column))
    else
        error("Invalid scaling mode")
    end

    return scaled_column
end

function get_forward_features_labels(features, labels, lookahead)
    n_rows, n_features = size(features)
    n_examples = div(n_rows, lookahead)

    forward_features = zeros(Float64, n_examples, n_features * lookahead)
    forward_labels = zeros(Float64, n_examples, lookahead)

    for ex in 1:n_examples
        for la in 1:lookahead
            forward_labels[ex, la] = labels[(lookahead * (ex - 1)) + la]
            for feat in 1:n_features
                forward_features[ex, (n_features * (la - 1)) + feat] = features[(lookahead * (ex - 1)) + la, feat]
            end
        end
    end

    return forward_features, forward_labels
end

function preprocess_data(data_dict)
    loc_data = data_dict["loc_data"]
    scale_mode = data_dict["scale_mode"]
    scale_base = data_dict["scale_base"]
    cols_features = data_dict["feat_cols"]
    col_label_price = data_dict["col_label_price"]
    col_label_fc_price = data_dict["col_label_fc_price"]
    lookahead = data_dict["lookahead"]
    days_train = data_dict["days_train"]
    last_ex_test = data_dict["last_ex_test"]
    train_share = data_dict["train_share"]
    val_split_mode = data_dict["val_split_mode"]
    cols_no_centering = data_dict["cols_no_centering"]

    data_all = get_data_pandas(loc_data)
    #data_all_scaled = scale_df(data_all, scale_mode, base=scale_base)
    data_all_scaled, stdev_yhat = scale_df_new(data_all,scale_mode,cols_no_centering)
    data_dict["stdev_yhat"] = stdev_yhat

    features = select(data_all, cols_features)
    features = Matrix(features)
    scaled_features = select(data_all_scaled, cols_features)
    scaled_features = Matrix(scaled_features)
    prices = select(data_all,col_label_price)
    prices = Matrix(prices)
    forecasted_prices = select(data_all,col_label_fc_price)
    forecasted_prices = Matrix(forecasted_prices)
    forecasted_prices_scaled = select(data_all_scaled,col_label_fc_price)
    forecasted_prices_scaled = Matrix(forecasted_prices_scaled)



    features_for_training=nothing
    if scale_mode == "none"
        features_for_training = features
    else
        features_for_training = scaled_features
    end

    #@show(size(features_for_training))

    forward_features, forward_prices = get_forward_features_labels(features_for_training, prices, lookahead)
    _, forward_fc_price = get_forward_features_labels(features_for_training, forecasted_prices, lookahead)
    _, forward_fc_price_scaled = get_forward_features_labels(features_for_training, forecasted_prices_scaled, lookahead)

    tot_n_examples = size(forward_features, 1)
    indices_train, indices_val, indices_test = get_indices(last_ex_test, days_train, train_share, tot_n_examples,val_split_mode)

    @show(indices_train)
    @show(indices_val)
    @show(indices_test)


    features_train = forward_features[indices_train, :]
    features_val = forward_features[indices_val, :]
    features_test = forward_features[indices_test, :]

    price_train = forward_prices[indices_train, :]
    price_val = forward_prices[indices_val, :]
    price_test = forward_prices[indices_test, :]

    price_fc_train = forward_fc_price[indices_train, :]
    price_fc_val = forward_fc_price[indices_val, :]
    price_fc_test = forward_fc_price[indices_test, :]

    price_fc_train_scaled = forward_fc_price_scaled[indices_train, :]
    price_fc_val_scaled = forward_fc_price_scaled[indices_val, :]
    price_fc_test_scaled = forward_fc_price_scaled[indices_test, :]



    return features_train, features_val, features_test, price_train, price_val, price_test, [price_fc_train,price_fc_val, price_fc_test], [price_fc_train_scaled,price_fc_val_scaled,price_fc_test_scaled]

    


end

function preprocess_labels(type,price_train,price_val,price_test,OP_params_dict,data_dict)

    if data_dict["scale_price"]
        price_train = price_train./data_dict["stdev_yhat"]
        price_val = price_val./data_dict["stdev_yhat"]
        price_test = price_test./data_dict["stdev_yhat"]
    end

    if type == "price"
        labels_train = price_train
        labels_val = price_val
        labels_test = price_test
    else
        sched_train, sched_val = retrieve_optimal_schedule_train_test(
            price_train,
            price_val,
            OP_params_dict,
        )
        labels_train = [price_train, sched_train]
        labels_val = price_val
        labels_test = price_test
    end

    return labels_train,labels_val,labels_test
end

function get_loc_fc(data_dict,training_dict,machine)
    
    dirname = nothing
    act = training_dict["activation_function"]
    if machine == "local"
        dirname = "./data/pretrained_fc/"
    elseif machine == "vsc"
        dirname = "../input_data/pretrained_fc/"
    end

    scaled_string = "unscaled"
    if data_dict["scale_price"]
        scaled_string = "scaled"
    end

    features_string = nothing
    if data_dict["feat_cols"] == ["weekday", "NL+FR","GEN_FC","y_hat"]
        features_string = "wd_nlfr_genfc_yhat"
    end

    filename = "$(dirname)model_$(act)_$(features_string)_$(scaled_string).bson"

    return filename

end

function get_type_mode_mu(reforecast_type,nn_type)
    train_mode = nothing
    if nn_type == "linear"
        train_mode = "linear"
    elseif nn_type == "softplus"
        train_mode = "nonlinear"
    end
    if reforecast_type == "Sp_SG"
        return "SG", train_mode, "auto"
    elseif reforecast_type == "Sp_IP"
        return "IP", train_mode, "auto"
    elseif reforecast_type == "Sp_IPs"
        return "IP", train_mode, "manual_s"
    elseif reforecast_type == "Sp_IPd"
        return "IP", train_mode, "manual_d"
    end
end



##### FUNCTIONS FOR SAVING THE OUTPUT OF THE TRAINING PROCEDURE #####

function save_df(dict_outcome, dir)
    path = joinpath(dir, "outcome.csv")

    # Sort the keys of the dictionary
    sorted_keys = sort(string.(keys(dict_outcome)))

    # Create an empty DataFrame
    df = DataFrame()

    # Add columns to the DataFrame in sorted order
    for key in sorted_keys
        df[!, Symbol(key)] = dict_outcome[key]
    end

    # Write the DataFrame to a CSV file
    CSV.write(path, df)
    println("Saved")
end

function save_outcome_par(list_outcome_dicts,dir)
    
    for outcome_dict in list_outcome_dicts
        list_best = outcome_dict["net_best"]
        list_opti = outcome_dict["net_opti"]
        hp_config = outcome_dict["a_config"]
        dict_evols = outcome_dict["dict_evols"]
        list_fc = outcome_dict["list_fc_lists"]

        save("$(dir)config_$(hp_config)_best_net.jld2","best_net", list_best)
        save("$(dir)config_$(hp_config)_opti_net.jld2","opti_net", list_opti)
        save("$(dir)config_$(hp_config)_train_evols.jld2","train_evols", dict_evols)
        save("$(dir)config_$(hp_config)_fc_evol.jld2","fc_evol",list_fc)       
    end

    dict_lists = convert_list_dict(list_outcome_dicts)

    pop!(dict_lists,"dict_evols")
    pop!(dict_lists,"list_fc_lists")
    pop!(dict_lists,"net_best")
    pop!(dict_lists,"net_opti")


    save_df(dict_lists,dir)

end

function convert_list_dict(list_dicts)
    #Function that converts list of dicts to dict of lists
    
    #Create empty dict with correct keys
    dict_lists = Dict()

    for key in keys(list_dicts[1])
        dict_lists[key] = []
    end

    #iteratively push values to correct list
    for config in 1:length(list_dicts)
        for key in keys(list_dicts[1])
            push!(dict_lists[key],list_dicts[config][key])
        end
    end

    return dict_lists
end




