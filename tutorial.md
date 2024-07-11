# How to run

This code is a modification of the original one, which specifically change the trained weights according to the hardware-fault behaviour. There are two main covered faults: `flip-bit` and `stuck-bit`; There are also two modes: `remove-bit` and `normal`. To modify the weight, use `modify_snn_weight_v2.py`.

1. Step 0: Download the pretrained weights, following the instruction from README.md.

2. Step 0.5: One can try to train the weights by oneself. The command to train can be seen in `script.sh`. However, this is not recommended for those who don't have the insight into training models. Also, the training takes a very long time.

3. Step 1: With the pretrained weights, one can start modifying these weights according to the hardware-fault behaviors.

```bash:
python3 modify_snn_weight_v2.py
```

4. Step 2: Changing the input parameters. There are six parameters: `fp_type`, `fixed_bits`, `error_type`, `frate`, `start_bit`, and `end_bit`.

* `fp_type` : Define the name of the output modified weights.
* `fixed_bits` : Define the number of bits of the modified weights. From float to fixed point numbers.
* `error-type` : Chose the type of faults, modes to modify the pretrained weights.
* `frate` : Define the probability of faults appearing in the pretrained weights.
* `start-bit`, `end-bit` : Define the range of bits that faults will appear in the pretrained weights.

For example:

```python:
    fp_type = 'fp16'
	fixed_bits = 16
    # error_type = 'remove_bits'
    error_type = 'stuck_bits'
    # error_type = 'flip_bits'
    # error_type = ''
    frate = 0.001
    start_bit = 12
    end_bit = 15
```
3. Step 3: Load the pretrained weights.

```python:
    model = VGG_SNN_STDB(vgg_name = 'VGG16',
                         activation = 'Linear',
                         labels = 10,
                         timesteps = 200,
                         leak = 1.0,
                         default_threshold = 1.0,
                         alpha = 0.3,
                         beta = 0.01,
                         dropout = 0.3,
                         kernel_size = 3,
                         dataset='CIFAR10')
    model = nn.DataParallel(model)
    pretrained_snn = './trained_models/snn/snn_vgg16_cifar10.pth'
    state = torch.load(pretrained_snn, map_location='cpu')
```

4. Step 4: Change the float pretrained weights into the fixed-point numbers to match the hardware behavior. Please refer to the respective function for more details.

```python:
	state = float_to_fix_point(state, key, fixed_bits)
```

5. Step 5: Insert the faults according to the pre-defined parameters in `step 2`. Please refer to the respective function for more details.

```python:
	state = insert_faults(state, key, frate, error_type, start_bit, end_bit)
```

6. Step 6: Save the modified weights.

```python:
    filename = './trained_models/snn/snn_vgg16_cifar10_'+fp_type+error_type+'.pth'
    torch.save(state, filename)
```

7. Step 7: Get the accuracy of the new modified weights. Use the `script.sh` to run. Change the input parameters according to your expected outcomes. Importantly,one needs to change the value of `--pretrained_snn` matching to the new modified weights.


