import jax
import jax.numpy as jnp
import numpy as np
import os
import pickle
from nnsmith.materialize.jax import JAXModel

def load_and_verify_nan(directory="nan_scenario_jax"):
    if not os.path.exists(directory):
        print(f"Error: Directory '{directory}' does not exist.")
        return

    model_path = os.path.join(directory, "model")
    if not os.path.exists(model_path):
        print(f"Error: Model directory '{model_path}' does not exist.")
        return

    try:
        model = JAXModel.load(model_path)
        print("Model loaded successfully.")
        print("Model structure:")
        print(model.ir)
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return

    inputs_outputs_path = os.path.join(directory, "inputs_outputs.pkl")
    if not os.path.exists(inputs_outputs_path):
        print(f"Error: Inputs and outputs file '{inputs_outputs_path}' does not exist.")
        return

    try:
        with open(inputs_outputs_path, "rb") as f:
            data = pickle.load(f)
        print("Inputs and outputs loaded successfully.")
    except Exception as e:
        print(f"Error loading inputs and outputs: {str(e)}")
        return

    inputs = {k: jnp.array(v) for k, v in data["inputs"].items()}
    expected_outputs = data["outputs"]

    # Run the model
    try:
        outputs = model.run(inputs)
        print("Model executed successfully.")
    except Exception as e:
        print(f"Error running the model: {str(e)}")
        print("Model input specs:")
        for name, spec in model.input_specs.items():
            print(f"  {name}: shape={spec.shape}, dtype={spec.dtype}")
        print("Provided inputs:")
        for name, array in inputs.items():
            print(f"  {name}: shape={array.shape}, dtype={array.dtype}")
        return
    
    print("\nInputs:")
    for name, array in inputs.items():
        print(f"{name}:")
        print(array)

    print("\nIntermediate and Final Outputs:")
    for name, array in outputs.items():
        print(f"{name}:")
        print(array)
    

    # Check for NaNs
    nan_found = False
    for name, array in outputs.items():
        if jnp.isnan(array).any():
            print(f"NaN found in output '{name}'")
            nan_found = True
            break

    if nan_found:
        print("Verification successful: NaN reproduced")
    else:
        print("Verification failed: No NaN found in the outputs")

    # Compare with saved outputs
    print("\nComparing with saved outputs:")
    for name, expected_array in expected_outputs.items():
        if name not in outputs:
            print(f"{name}: Not found in current outputs")
            continue
        actual_array = np.array(outputs[name])  # Convert to numpy for comparison
        if np.array_equal(actual_array, expected_array):
            print(f"{name}: Matches exactly")
        else:
            print(f"{name}: Differs")
            print("  Expected:", expected_array.flatten()[:5])
            print("  Actual:  ", actual_array.flatten()[:5])

if __name__ == "__main__":
    jax.config.update('jax_platform_name', 'cpu')  # Use CPU for consistency
    load_and_verify_nan()