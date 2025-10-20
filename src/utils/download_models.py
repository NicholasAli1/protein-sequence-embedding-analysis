"""
Download ProtBERT model with retry logic.
"""
import time
from transformers import BertModel, BertTokenizer

def download_protbert():
    """Download ProtBERT model with retries."""
    max_retries = 3
    retry_delay = 5
    
    for attempt in range(max_retries):
        try:
            print(f"\n🔄 Attempt {attempt + 1}/{max_retries}: Downloading ProtBERT...")
            
            # Download tokenizer
            print("  → Downloading tokenizer...")
            tokenizer = BertTokenizer.from_pretrained(
                "Rostlab/prot_bert", 
                do_lower_case=False,
                resume_download=True
            )
            print("  ✅ Tokenizer downloaded")
            
            # Download model
            print("  → Downloading model (this may take 5-10 minutes)...")
            model = BertModel.from_pretrained(
                "Rostlab/prot_bert",
                resume_download=True
            )
            print("  ✅ Model downloaded successfully!")
            
            return True
            
        except Exception as e:
            print(f"  ❌ Attempt {attempt + 1} failed: {str(e)[:100]}")
            if attempt < max_retries - 1:
                print(f"  ⏳ Waiting {retry_delay} seconds before retry...")
                time.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff
            else:
                print("\n❌ All download attempts failed!")
                return False
    
    return False

if __name__ == "__main__":
    print("="*70)
    print("ProtBERT Model Downloader")
    print("="*70)
    print("\nThis will download the ProtBERT model (~1.6 GB)")
    print("Download may take 5-15 minutes depending on connection speed.\n")
    
    success = download_protbert()
    
    if success:
        print("\n" + "="*70)
        print("✅ SUCCESS! ProtBERT model is ready to use.")
        print("="*70)
        print("\nYou can now run: python3 run_pipeline.py")
    else:
        print("\n" + "="*70)
        print("❌ DOWNLOAD FAILED")
        print("="*70)
        print("\nTroubleshooting:")
        print("  • Check your internet connection")
        print("  • Try again in a few minutes (HuggingFace servers may be busy)")
        print("  • Run this script again: python3 download_models.py")
