#!/usr/bin/env bash

echo "run model with cifar-10"
echo "inception"
optirun python feature_extraction.py --training_file inception_cifar10_100_bottleneck_features_train.p --validation_file inception_cifar10_bottleneck_features_validation.p

echo "\nresnet"
optirun python feature_extraction.py --training_file resnet_cifar10_100_bottleneck_features_train.p --validation_file resnet_cifar10_bottleneck_features_validation.p

echo "\nvgg"
optirun python feature_extraction.py --training_file vgg_cifar10_100_bottleneck_features_train.p --validation_file vgg_cifar10_bottleneck_features_validation.p

echo "run model with traffic-sign"
echo "inception"
optirun python feature_extraction.py --training_file inception_traffic_100_bottleneck_features_train.p --validation_file inception_traffic_bottleneck_features_validation.p

echo "\nresnet"
optirun python feature_extraction.py --training_file resnet_traffic_100_bottleneck_features_train.p --validation_file resnet_traffic_bottleneck_features_validation.p

echo "\nvgg"
optirun python feature_extraction.py --training_file vgg_traffic_100_bottleneck_features_train.p --validation_file vgg_traffic_bottleneck_features_validation.p

