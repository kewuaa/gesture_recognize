---@diagnostic disable: undefined-global
add_rules("mode.debug", "mode.release")
add_requires("python", "fftw", {system = true})
add_requires("opencv", {
    configs = {
        gtk = true,
        bundled = false,
        shared = true
    },
})

target("gesture_recognize")
    set_kind("binary")
    add_packages("opencv", "python", "fftw")
    add_files("src/*.cpp")
    add_defines("NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION")

    on_load(function(target)
        local numpy_include = os.iorun("python -c 'import numpy as np; print(np.get_include(), end=\"\")'")
        target:add("includedirs", numpy_include)
    end)
