import jax
import jax.numpy as jnp
import numpy as np
import os
import pickle
from nnsmith.backends import BackendFactory
from nnsmith.graph_gen import model_gen
from nnsmith.materialize import Model
from nnsmith.narrow_spec import auto_opset
from nnsmith.abstract.dtype import DType
from nnsmith.materialize.jax import JAXModel

def generate_jax_model():
    # Initialize the backend factory
    factory = BackendFactory.init("jax_xla", target="cpu", optmax=False)

    # Initialize the model type
    ModelType = Model.init("jax", backend_target="cpu")

    # Get the opset and exclude problematic operations
    opset = auto_opset(ModelType, factory)
    excluded_ops = ['Cholesky', 'Eigh', 'Reverse']  # Add any other problematic ops here
    filtered_opset = [op for op in opset if op.__name__ not in excluded_ops]

    # Generate the model
    gen = model_gen(
        opset=filtered_opset,
        seed=np.random.randint(0, 10000),  # Random seed for each generation
        max_nodes=5,  # Adjust this for larger or smaller models
        dtype_choices=[DType.float32, DType.float64]  # Exclude half-precision floats
    )

    # Create a concrete model from the generated graph
    model = ModelType.from_gir(gen.make_concrete())
    return model

def fuzz_model(model, max_iterations=1000):
    for i in range(max_iterations):
        # Generate random inputs
        random_inputs = model.random_inputs()

        # Run the model with random inputs
        try:
            outputs = model.run(random_inputs)

            # Check for NaNs in the outputs
            for name, tensor in outputs.items():
                if jnp.isnan(tensor).any():
                    print(f"NaN found in output '{name}' at iteration {i+1}")
                    print("Input shapes:")
                    for in_name, in_tensor in random_inputs.items():
                        print(f"{in_name}: {in_tensor.shape}")
                    print("Output shapes:")
                    for out_name, out_tensor in outputs.items():
                        print(f"{out_name}: {out_tensor.shape}")
                    return True, random_inputs, outputs

        except Exception as e:
            print(f"Error at iteration {i+1}: {str(e)}")

    print("No NaNs found after", max_iterations, "iterations")
    return False, None, None

def save_nan_scenario(model, inputs, outputs, directory="nan_scenario_jax"):
    os.makedirs(directory, exist_ok=True)
    
    # Save the model
    model.dump(os.path.join(directory, "model"))
    
    # Save inputs and outputs
    with open(os.path.join(directory, "inputs_outputs.pkl"), "wb") as f:
        pickle.dump({
            "inputs": {k: np.array(v) for k, v in inputs.items()},
            "outputs": {k: np.array(v) for k, v in outputs.items()}
        }, f)

    print(f"NaN scenario saved in directory: {directory}")

def main():
    np.random.seed(42)
    jax.config.update('jax_platform_name', 'cpu')  # Use CPU for consistency

    while True:
        print("Generating new JAX model...")
        model = generate_jax_model()

        print("\nModel structure:")
        print(model.ir)

        print("\nFuzzing the model...")
        nan_found, inputs, outputs = fuzz_model(model)

        if nan_found:
            print("\nNaN found! Saving the scenario...")
            save_nan_scenario(model, inputs, outputs)
            break
        else:
            print("No NaNs found for this model. Generating a new model...")

if __name__ == "__main__":
    main()