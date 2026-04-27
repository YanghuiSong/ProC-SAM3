import torch
import open_clip

class RemoteCLIPModel:
    """
    RemoteCLIP model for semantic alignment between text and visual features
    Based on RemoteCLIP: A Vision Language Foundation Model for Remote Sensing
    """

    def __init__(self, ckpt_path, model_name="ViT-L-14"):
        """
        Initialize the RemoteCLIP model
        
        Args:
            ckpt_path: Path to the checkpoint file
            model_name: Name of the model architecture (default: ViT-L-14)
        """
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(model_name)
        
        # Load checkpoint
        if ckpt_path:
            ckpt = torch.load(ckpt_path, map_location="cpu")
            self.model.load_state_dict(ckpt)
        
        self.model = self.model.cuda().eval()
        self.tokenizer = open_clip.get_tokenizer(model_name)

    def encode_text(self, texts):
        """
        Encode text prompts to feature vectors
        
        Args:
            texts: List of text prompts or single text string
            
        Returns:
            Normalized text features
        """
        if isinstance(texts, str):
            texts = [texts]
        
        tokens = self.tokenizer(texts)
        tokens = tokens.cuda()
        
        with torch.no_grad():
            text_features = self.model.encode_text(tokens)
        
        # Normalize features
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return text_features

    def encode_image(self, images):
        """
        Encode images to feature vectors
        
        Args:
            images: Preprocessed image tensors
            
        Returns:
            Normalized image features
        """
        with torch.no_grad():
            image_features = self.model.encode_image(images)
        
        # Normalize features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        return image_features