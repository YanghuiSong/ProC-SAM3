def load_class_names(path):
    """
    Load class names from a text file.
    
    Args:
        path (str): Path to the class names file
        
    Returns:
        List[str]: List of class names
    """
    with open(path, 'r') as f:
        lines = f.readlines()
    
    class_names = []
    for line in lines:
        line = line.strip()
        if line:  # Skip empty lines
            class_names.append(line)
    
    return class_names