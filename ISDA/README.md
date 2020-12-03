**An Implement of paper "An Improved SDA Based Defect Prediction Framework for Both Within-Project and Cross-Project Class-Imbalance Problems” (IEEE 2017)** 

- [x] `ISDA`-Based Within-Project Prediction
  * according the paper, I get the projective transformation matrix V with complex numbers. However, the random forest classifier cannot support complex type. So, I choose the real part of the matrix V. I think something is wrong, but I don’t know how to tackle it.
  * refer to paper "Subclass Discriminant Analysis” (IEEE 2006)
- [x]  `SSTCA`+`ISDA` for Cross-Project Prediction
  - [ ] Use `TCA`+`ISDA` now, need to change `TCA` to `SSTCA`
  - refer to paper "Domain Adaptation via Transfer Component Analysis” (IEEE 2011)

