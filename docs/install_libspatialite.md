# Installing SpatiaLite 5.1.0+

- We use [VirtualKNN2](https://www.gaia-gis.it/fossil/libspatialite/wiki?name=KNN2), which was added in [check-in 03786a62cd](https://www.gaia-gis.it/fossil/libspatialite/info/03786a62cdb4ab17) of SpatialLite.
- Download and build SpatiaLite:
  - Make sure you have [fossil](https://www.fossil-scm.org) installed
  - Clone the libspatialite repository
    ```
    fossil clone https://www.gaia-gis.it/fossil/libspatialite
    cd libspatialite
    ```
  - If you have an older version of fossil, you might have to do clone and then open the repository:
    ```
    fossil clone https://www.gaia-gis.it/fossil/libspatialite libspatialite.fossil
    mkdir libspatialite && cd libspatialite
    fossil open ../libspatialite.fossil
    ```
  - Install libspatialite
    ```
    # you can build with freexl if you like, but it isn't needed
    ./configure --disable-freexl
    make
    sudo make install
    ```
- Adding your newly-built mod_spatialite.so to `config.ini`. Instructions for doing so can be found in [README.md](/README.md).
