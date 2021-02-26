# RL Policies

The code and weights in policy folder can be saved with CtF folder.

## 1. A3C (TensorFlow)

### Specification

- Tensorflow 1.12

### Contents

- policy_A3C.py
- A3C_model (folder)

### Example usage

``` py
import policy.policy_A3C
...
policy_blue = policy.policy_A3C.PolicyGen(env.get_map, env.get_team_blue)
```

# Note:

Only upload the final weight and policy file in order to save repository space.

