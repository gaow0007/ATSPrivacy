from autoaugment import SubPolicy

policies = [
        SubPolicy(0.1, "invert", 7),
        SubPolicy(0.2, "contrast", 6),
        SubPolicy(0.7, "rotate", 2),
        SubPolicy(0.3, "translateX", 9),
        SubPolicy(0.8, "sharpness", 1),

        SubPolicy(0.9, "sharpness", 3),
        SubPolicy(0.5, "shearY", 2),
        SubPolicy(0.7, "translateY", 2) ,
        SubPolicy(0.5, "autocontrast", 5),
        SubPolicy(0.9, "equalize", 2), #

        SubPolicy(0.2, "shearY", 5),
        SubPolicy(0.3, "posterize", 5), #
        SubPolicy(0.4, "color", 3),
        SubPolicy(0.6, "brightness", 5), #
        SubPolicy(0.3, "sharpness", 9),

        SubPolicy(0.7, "brightness", 9),
        SubPolicy(0.6, "equalize", 5),
        SubPolicy(0.5, "equalize", 1),
        SubPolicy(0.6, "contrast", 7),
        SubPolicy(0.6, "sharpness", 5),
        
        SubPolicy(0.7, "color", 5),
        SubPolicy(0.5, "translateX", 5), #
        SubPolicy(0.3, "equalize", 7),
        SubPolicy(0.4, "autocontrast", 8),
        SubPolicy(0.4, "translateY", 3),
        SubPolicy(0.2, "sharpness", 6),
        SubPolicy(0.9, "brightness", 6),
        SubPolicy(0.2, "color", 8),
        SubPolicy(0.5, "solarize", 0),
        SubPolicy(0.0, "invert", 0), #
        SubPolicy(0.2, "equalize", 0),
        SubPolicy(0.6, "autocontrast", 0), #
        SubPolicy(0.2, "equalize", 8),
        SubPolicy(0.6, "equalize", 4),
        SubPolicy(0.9, "color", 5),
        SubPolicy(0.6, "equalize", 5), #
        SubPolicy(0.8, "autocontrast", 4),
        SubPolicy(0.2, "solarize", 4), #
        SubPolicy(0.1, "brightness", 3),
        SubPolicy(0.7, "color", 0),
        SubPolicy(0.4, "solarize", 1),
        SubPolicy(0.9, "autocontrast", 0), #
        SubPolicy(0.9, "translateY", 3),
        SubPolicy(0.7, "translateY", 3), #
        SubPolicy(0.9, "autocontrast", 1),
        SubPolicy(0.8, "solarize", 1), #
        SubPolicy(0.8, "equalize", 5),
        SubPolicy(0.1, "invert", 0),  #
        SubPolicy(0.7, "translateY", 3),
        SubPolicy(0.9, "autocontrast", 1),
        ]

if __name__ == '__main__':
        print(len(policies))
