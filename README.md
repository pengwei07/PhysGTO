# PhysGTO

PhysGTO: An Efficient Graph-Transformer Operator for Learning Physical Dynamics with Manifolds Embedding

Traditional numerical solvers for physical simulations face significant computational burdens across complex, dynamic scenarios. To overcome these challenges, we propose PhysGTO, an efficient Graph-Transformer Operator characterized by:

- Unified Graph Embedding: **Aligns heterogeneous conditions and samples topology-aware graphs for structure-preserving discretization.**

- Flux-Oriented Message **Passing: Captures fine-grained local dynamics with lightweight computation.**

- Projection-Inspired **Attention: Models global dependencies efficiently with linear complexity.**

- Manifold Embedding: **Enhances geometric adaptability and cross-scenario generalization.**

- State-of-the-Art Performance: **Achieves leading accuracy and major computational savings across eleven diverse unstructured mesh benchmarks.**


## Get Started

Our model is designed to address three distinct scenarios:

- Task 1: Learning complex physical patterns on unstructured meshes.

- Task 2: Reliable long-term dynamics forecasting.

- Task 3: Scaling to large-scale 3D geometries.

At present, we demonstrate one representative case for each task.

 Please maintain the file structure shown below to run the script by default：

```sh
The project
|
└─── code
|    └─── task1
|          └─── src
|               ...
|          main.py
|    └─── task2
|          └─── config
|               ...
|          └─── src
|               ...
|          main.py
|          train.sh
|    └─── task3
|          └─── config
|               ...
|          └─── src
|               ...
|          main.py
|          train.sh
|
└─── data
|    └─── task1
|          ...
|    └─── task2
|          ...
|    └─── task3
|          ...
```


Add dataset folder if it does not exist, add data to corresponding dataset. You should change the environment settings in the file according to your own hardware configuration.



**Data Format:**

The geometry format of our dataset (PhysGTO_Dataset) should be as follows:

```python
Node_pos = [
    [X1, Y1, Z1],
    [X2, Y2, Z2],
   ...
]
Cells = [
    [p1, p2, p3],
    [p4, p5, p6],
   ...
]
```
- **X,Y,Z**: (N_points x 3) numpy array(2D or 3D), representing input mesh points. X_dim, Y_dim, Z_dim are input dimensions of geometry.

- **Cells**: (N_cells x 3) numpy array, representing Geometric Cells, which contains three points. The minimum index is `0`, and the maximum value is `N_points-1`.

- **Note**: <br />
    **I.** For a single sample, The number of points must match, i.e, ``X.shape[0]=Y.shape[0]``, but it can vary with different samples. <br />
    **II.** If additional preprocessing is required for the data, please modify it in the corresponding dataset file in the directory `code/src`

The field input should be as follows:

```python
input_field1 = [
    [a1, a2, a3...],
    [a4, a5, a6...],
   ...
]
input_field2 = [
    [b1, b2, b3...],
    [b4, b5, b6...],
   ...
]
...
```
- **input_field**: (N_points x C) numpy array, representing the input fields. 

input mesh points. X_dim, Y_dim, Z_dim are input dimensions of geometry.

- **Note**: The input fields in our framework are flexible and task-dependent: they can be absent or numerous depending on the problem setting. <br />
For example, in 3D vehicle pressure estimation, no additional input fields are required beyond the geometry itself;In unstructured mesh problems, the inputs may include initial states, various physical fields, or other relevant physical information;For long-term dynamics forecasting, the inputs typically consist of the initial conditions, node types, and other auxiliary condition fields. <br />
Additionally, for temporal problems, we incorporate **explicit time encoding** to further enhance model performance.
This flexible input design significantly improves the model’s ability to adapt and generalize across a wide range of complex scenarios.





## Requirements

- torch==2.1.0
- torch_scatter==2.1.0
- numpy==1.24.3
- pandas==2.0.1
- plyfile==0.7.4
- h5py==3.9.0
- vtk==9.2.6
- tensorboardX==2.6


## Contact

If you have any questions or want to use the code, please contact [liupw@zju.edu.cn](mailto:liupw@zju.edu.cn).

## Contributing

We welcome contributions to improve the dataset or project. Please submit pull requests for review.

