# dFUSE: Differentiable Framework for Understanding Structural Errors

**Version 0.2.10** - A GPU-native, differentiable implementation of the FUSE hydrological model framework in C++, with Sundials and Enzyme. 

**Note dFUSE is in active development, expect to find broken or unfinished code**

Based on [Clark et al. (2008) "Framework for Understanding Structural Errors (FUSE): A modular framework to diagnose differences between hydrological models"](http://dx.doi.org/10.1029/2007WR006735), Water Resources Research.


### Differentiability

All physics computations use smooth approximations for discontinuities (following Kavetski & Kuczera 2007):

```cpp
// Logistic smoothing for bucket overflow
Real logistic_overflow(Real S, Real S_max, Real w) {
    Real x = (S - S_max - w * 5) / w;
    return 1.0 / (1.0 + exp(-x));
}
```


## References

- Clark, M. P., et al. (2008). Framework for Understanding Structural Errors (FUSE): A modular framework to diagnose differences between hydrological models. Water Resources Research, 44(12). [doi:10.1029/2007WR006735](http://dx.doi.org/10.1029/2007WR006735)

- Henn, B., et al. (2015). An assessment of differences in gridded precipitation datasets in complex terrain. Journal of Hydrology, 530, 167-180.

- Kavetski, D., & Kuczera, G. (2007). Model smoothing strategies to remove microscale discontinuities and spurious secondary optima in objective functions in hydrological calibration. Water Resources Research, 43(3).

## License

GNU General Public License v3.0 (same as original FUSE)

## Contributing

Contributions welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## Acknowledgments

- Original FUSE implementation by Martyn Clark and collaborators at NCAR
- Clark et al. (2008) for the modular modeling framework design
- Enzyme AD project for automatic differentiation support
