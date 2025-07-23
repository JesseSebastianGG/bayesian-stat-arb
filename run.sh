#!/bin/bash
docker build -t bayes-stat-arb .
docker run -p 8888:8888 -v $(pwd):/app bayes-stat-arb
