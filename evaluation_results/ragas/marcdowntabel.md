| experiment           | judge       |   faithfulness |   answer_relevancy |   answer_correctness |   context_precision |   context_recall |   context_sufficiency | framework   |
|:---------------------|:------------|---------------:|-------------------:|---------------------:|--------------------:|-----------------:|----------------------:|:------------|
| 01_ollama_default_20 (basic rag) | gpt-4o-mini |        0.92235 |            0.82765 |             0.590475 |            0.867715 |           0.8875 |                   0.6 | ragas       |
| **01_ollama_default_20 ((basic rag)  - rerun)** | gpt-4o-mini |       0.912765 |            0.82272 |              0.56959 |            0.867715 |            0.925 |                   0.6 | ragas       |
| **01_ollama_default_config_results (100)** | gpt-4o-mini |       0.876011 |           0.776433 |             0.599567 |            0.837644 |         0.835003 | no results | ragas
| 01_ollama_default_20 (basic rag) | gpt-5.1 |       0.851735 |            0.88097 |             0.461595 |            0.646945 |             0.66 |                   0.1 | ragas       |
| **11_ollama_adaptive_20 (version 0)** | gpt-4o-mini |       0.820155 |            0.62249 |              0.55456 |             0.90573 |           0.8625 |                  0.55 | ragas       |
| 11_ollama_adaptive_20 (version 0)| gpt-5.1 |        0.66964 |            0.84757 |             0.475175 |             0.57722 |              0.6 |                  0.25 | ragas       |
| gpt4omini_exp_0001_results | gpt-4o-mini |        0.73533 |            0.73501 |              0.58112 |            0.956945 |         0.883335 |                   0.6 | ragas       |
| gpt41nano_exp_0002_results | gpt-4o-mini |         0.4729 |            0.40095 |              0.47627 |                0.75 |         0.633335 |                   0.3 | ragas       |
| gpt51_exp_0000_results | gpt-4o-mini |        0.72634 |           0.426715 |             0.453505 |            0.943335 |         0.945835 |                  0.45 | ragas       |
| new_adaptive_v1_results | gpt-4o-mini |       0.784445 |           0.545055 |              0.52379 |            0.522495 |            0.875 |                   0.4 | ragas       |
| new_adaptive_v2_results | gpt-4o-mini |       0.772305 |           0.668335 |              0.51605 |            0.535965 |          0.91667 |                   0.4 | ragas       |
| new_adaptive_v3_results | gpt-4o-mini |        0.83028 |           0.532765 |              0.54454 |                0.85 |          0.74167 | no results |ragas       |
| new_adaptive_v4_results | gpt-4o-mini |       0.805795 |            0.70144 |             0.537045 |             0.74378 |         0.841665 | no results |ragas       |
| new_adaptive_v5_results | gpt-4o-mini |        0.69107 |            0.46545 |               0.4559 |            0.761935 |           0.7375 |                   0.2 | ragas       |
| new_adaptive_v6_results | gpt-4o-mini |       0.659265 |            0.51609 |             0.448205 |             0.68467 |           0.8125 |                   0.2 | ragas       |
| new_adaptive_v7_results | gpt-4o-mini |       0.845535 |            0.62139 |               0.5491 |             0.73331 |             0.85 |                  0.45 | ragas       |
| new_adaptive_v8_results | gpt-4o-mini |        0.87748 |            0.58385 |              0.51288 |             0.74167 |         0.808335 |                  0.45 | ragas       |
| new_adaptive_v8_results | gpt-5.1 |        0.63707 |           0.612015 |             0.448315 |             0.53861 |              0.6 |                  0.15 | ragas       |
| **new_adaptive_v9_results (20)** | gpt-4o-mini |       0.882753 |           0.703395 |             0.630589 |            0.843205 |         0.903511 |              0.631579 | ragas       |
| new_adaptive_v9_1_results | gpt-4o-mini |       0.918921 |           0.725929 |              0.57812 |            0.856165 |         0.855221 |              0.515152 | ragas       |
| incremental_adaptive_v11_results | gpt-4o-mini |       0.925755 |            0.85391 |             0.628855 |            0.844035 |              0.9 |                   0.6 | ragas       |
| incremental_adaptive_v12_results | gpt-4o-mini |        0.93199 |            0.81404 |              0.61612 |            0.826045 |            0.925 |                   0.6 | ragas       |
| incremental_adaptive_v13_results | gpt-4o-mini |        0.90434 |            0.85153 |             0.587979 |            0.816255 |         0.895835 |                  0.55 | ragas       |
| incremental_adaptive_v14_results | gpt-4o-mini |        0.89564 |            0.71749 |              0.63492 |             0.77105 |         0.895835 |                   0.6 | ragas       |
| **incremental_adaptive_v15_results (20)** | gpt-4o-mini |        0.90265 |           0.775925 |              0.62844 |            0.844035 |         0.958335 |                  0.55 | ragas       |
| incremental_adaptive_v15_results | gpt-5.1 |       0.807115 |           0.899995 |             0.494405 |             0.68111 |           0.6875 |                   0.3 | ragas       |
| incremental_adaptive_v15_results (100) | gpt-4o-mini |       0.899197 |            0.55739 |             0.589222 |            0.834855 |         0.827384 |              0.459184 | ragas       |
| incremental_adaptive_v15_1_results (20) | gpt-4o-mini |       0.937195 |           0.873853 |             0.624115 |            0.857925 |           0.9125 |                   0.6 | ragas       |
| incremental_adaptive_v15_1_results (100) | gpt-4o-mini |        0.90891 |           0.766859 |             0.597341 |            0.874037 |         0.850171 |              0.505051 | ragas       |
| incremental_adaptive_v15_2_results (20) | gpt-4o-mini |        0.92481 |            0.88381 |             0.614405 |             0.87188 |           0.9125 |                  0.55 | ragas       |
| incremental_adaptive_v15_2_results (20) | gpt-5.1 |        0.78786 |           0.931055 |             0.486715 |             0.62528 |            0.735 |                   0.4 | ragas       |
| incremental_adaptive_v15_2_results (100) | gpt-4o-mini |       0.908806 |           0.773871 |             0.577444 |            0.869393 |         0.842596 |              0.530612 | ragas       |
| incremental_adaptive_v16_results (20) | gpt-4o-mini |       0.858605 |            0.64164 |              0.58699 |             0.61493 |          0.89167 |                  0.45 | ragas       |
| 01_ollama_default_config_results (web) | gpt-4o-mini |       0.254545 |           0.335518 |             0.293918 |           0.0484818 |         0.272727 |              0.181818 | ragas       |
| web_incremental_adaptive_v17_results (web) | gpt-4o-mini |       0.683982 |             0.8509 |             0.569045 |            0.257573 |         0.545455 |              0.454545 | ragas       |
| incremental_adaptive_v17_results (20) | gpt-4o-mini |         0.9238 |            0.87838 |             0.644695 |            0.817715 |           0.9125 |                   0.6 | ragas       |
| incremental_adaptive_v17_results (100) | gpt-4o-mini |       0.917582 |           0.790491 |             0.589979 |            0.860541 |         0.853538 |              0.535354 | ragas       |
| web_incremental_adaptive_v18_results (web) | gpt-4o-mini |       0.693718 |             0.8292 |             0.538736 |            0.446973 |         0.454545 |              0.454545 | ragas       |