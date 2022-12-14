#!bin/bash

if python3 -c 'import pkgutil; exit(not pkgutil.find_loader("tflite_runtime"))'; then
    echo 'Tflite was already installed'
    pip list | grep tflite-runtime #
else
    echo 'Installing Tflite....'
    python3 -m pip install tflite-runtime
    a=$(pip list | grep tflite-runtime)
    if  [ a ]; then
    echo "Tflite was successfully installed"
    printf "Version of tflite is: "
    pip list | grep tflite-runtime #
    else 
        echo "Tflite installation failed"
    fi 
fi




