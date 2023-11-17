module NeuralNetworkStructure

    struct NeuralNetwork
        weights::Array{Array{Float64, 2}, 1}  # Array of weight matrices for each layer
        biases::Array{Vector{Float64}, 1}     # Array of bias vectors for each layer
        activations::Array{Function, 1}       # Array of activation functions for each layer

        function NeuralNetwork(layer_sizes::Vector{Int}, activation_functions::Vector{Function})
            # Check for valid input
            if length(layer_sizes) < 2
                error("There should be at least two layers (input and output).")
            end
            if length(activation_functions) != length(layer_sizes) - 1
                error("The number of activation functions should be one less than the number of layers.")
            end

            # Initialize weights and biases
            weights = [randn(layer_sizes[i+1], layer_sizes[i]) for i in 1:length(layer_sizes)-1]
            biases = [randn(layer_sizes[i]) for i in 2:length(layer_sizes)]

            new(weights, biases, activation_functions)
        end
    end

    function (nn::NeuralNetwork)(x::Vector{Float64})
        for (i, (W, b)) in enumerate(zip(nn.weights, nn.biases))
            x = nn.activations[i](W * x .+ b)
        end
        return x
    end

end

module TrainingOutputStructure
    x=1
end

