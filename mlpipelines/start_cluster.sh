#!/bin/bash

# Start the cluster by scaling up the default node pool

# Cluster information
KFP_CLUSTER_NAME=kfp-cf
COMPUTE_ZONE=europe-west1-b

gcloud container clusters resize $KFP_CLUSTER_NAME \
  --zone $COMPUTE_ZONE \
  --node-pool default-pool \
  --num-nodes 3
