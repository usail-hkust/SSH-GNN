# Semi-Supervised Air Quality Forecasting via Self-Supervised Hierarchical Graph Neural Network

Predicting air quality in fine spatiotemporal granularity is of great importance for air pollution control and urban sustainability. However, existing studies are either focused on predicting station-wise future air quality, or inferring current air quality for unmonitored regions. How to accurately forecast future air quality for these unmonitored regions in a fine granularity remains an unexplored problem. In this paper, we propose the Self-Supervised Hierarchical Graph Neural Network (SSH-GNN), for fine-grained air quality forecasting in a semi-supervised way. Specifically, to augment spatially sparse air quality observations, SSH-GNN first approximates the city-wide air quality distribution based on historical readings and various urban contextual factors (e.g., weather conditions and traffic flows). Then, we propose a hierarchical recurrent graph neural network to make city-wide predictions, which encodes the spatial hierarchy of urban regions for long-range spatiotemporal correlation modeling. Moreover, by leveraging spatiotemporal self-supervision strategies, SSH-GNN exploits both universal topological and contextual patterns to further enhance the forecasting effectiveness. Extensive experiments on two real-world datasets show that SSH-GNN significantly outperforms the state-of-the-art algorithms.

This is a pytorch implementation of SSH-GNN model as described in the following paper: 
[Semi-Supervised Air Quality Forecasting via Self-Supervised Hierarchical Graph Neural Network, TKDE 2022].

## References
If you find the code useful for your research, please consider citing
```bib
@article{han2022semi,
  title={Semi-supervised air quality forecasting via self-supervised hierarchical graph neural network},
  author={Han, Jindong and Liu, Hao and Xiong, Haoyi and Yang, Jing},
  journal={IEEE Transactions on Knowledge and Data Engineering},
  year={2022}
}
```

## Acknowledgement
We thank the authors for the following repositories for code reference:
[Graph WaveNet](https://github.com/nnzhan/Graph-WaveNet), etc.
