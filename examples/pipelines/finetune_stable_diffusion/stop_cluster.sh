#!/bin/bash

# Stop the cluster by scaling down the default node pool

# Cluster information
KFP_CLUSTER_NAME=kfp-express
COMPUTE_ZONE=europe-west4-a

gcloud container clusters resize $KFP_CLUSTER_NAME \
  --zone $COMPUTE_ZONE \
  --node-pool default-pool \
  --num-nodes 0
