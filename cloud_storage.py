"""
Cloud storage integration for uploaded PDF files.
"""
import os
from typing import Optional, List
import io

# Optional imports for cloud storage
try:
    import boto3
    from botocore.exceptions import ClientError
    BOTO3_AVAILABLE = True
except ImportError:
    BOTO3_AVAILABLE = False
    ClientError = Exception


class CloudStorage:
    """Manages cloud storage for uploaded files."""
    
    def __init__(
        self,
        storage_type: str = "local",
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
        bucket_name: Optional[str] = None,
        region: str = "us-east-1"
    ):
        """
        Initialize cloud storage.
        
        Args:
            storage_type: "local", "s3", or "gcs"
            aws_access_key_id: AWS access key ID
            aws_secret_access_key: AWS secret access key
            bucket_name: S3 bucket name
            region: AWS region
        """
        self.storage_type = storage_type
        self.bucket_name = bucket_name or os.getenv("AWS_S3_BUCKET")
        self.local_storage_path = os.getenv("LOCAL_STORAGE_PATH", "uploads")
        
        if storage_type == "s3":
            if not BOTO3_AVAILABLE:
                raise ImportError("boto3 is required for S3 storage. Install it with: pip install boto3")
            self.s3_client = boto3.client(
                's3',
                aws_access_key_id=aws_access_key_id or os.getenv("AWS_ACCESS_KEY_ID"),
                aws_secret_access_key=aws_secret_access_key or os.getenv("AWS_SECRET_ACCESS_KEY"),
                region_name=region
            )
        elif storage_type == "gcs":
            # Google Cloud Storage would require google-cloud-storage library
            # For now, we'll use local storage as fallback
            self.storage_type = "local"
        
        # Create local storage directory if it doesn't exist
        if self.storage_type == "local":
            os.makedirs(self.local_storage_path, exist_ok=True)
    
    def upload_file(self, file_content: bytes, file_name: str, user_id: Optional[str] = None) -> str:
        """
        Upload a file to cloud storage.
        
        Args:
            file_content: File content as bytes
            file_name: Name of the file
            user_id: Optional user ID for organizing files
            
        Returns:
            File path/URL in storage
        """
        if user_id:
            storage_path = f"{user_id}/{file_name}"
        else:
            storage_path = file_name
        
        if self.storage_type == "s3":
            try:
                self.s3_client.put_object(
                    Bucket=self.bucket_name,
                    Key=storage_path,
                    Body=file_content
                )
                return f"s3://{self.bucket_name}/{storage_path}"
            except ClientError as e:
                raise Exception(f"Error uploading to S3: {str(e)}")
        else:  # local storage
            local_path = os.path.join(self.local_storage_path, storage_path)
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            
            with open(local_path, 'wb') as f:
                f.write(file_content)
            
            return local_path
    
    def download_file(self, file_path: str) -> bytes:
        """
        Download a file from cloud storage.
        
        Args:
            file_path: Path/URL of the file in storage
            
        Returns:
            File content as bytes
        """
        if self.storage_type == "s3":
            try:
                # Extract bucket and key from S3 path
                if file_path.startswith("s3://"):
                    parts = file_path.replace("s3://", "").split("/", 1)
                    bucket = parts[0]
                    key = parts[1] if len(parts) > 1 else ""
                else:
                    bucket = self.bucket_name
                    key = file_path
                
                response = self.s3_client.get_object(Bucket=bucket, Key=key)
                return response['Body'].read()
            except ClientError as e:
                raise Exception(f"Error downloading from S3: {str(e)}")
        else:  # local storage
            if os.path.exists(file_path):
                with open(file_path, 'rb') as f:
                    return f.read()
            else:
                raise FileNotFoundError(f"File not found: {file_path}")
    
    def delete_file(self, file_path: str) -> bool:
        """
        Delete a file from cloud storage.
        
        Args:
            file_path: Path/URL of the file in storage
            
        Returns:
            True if successful, False otherwise
        """
        if self.storage_type == "s3":
            try:
                if file_path.startswith("s3://"):
                    parts = file_path.replace("s3://", "").split("/", 1)
                    bucket = parts[0]
                    key = parts[1] if len(parts) > 1 else ""
                else:
                    bucket = self.bucket_name
                    key = file_path
                
                self.s3_client.delete_object(Bucket=bucket, Key=key)
                return True
            except ClientError:
                return False
        else:  # local storage
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
                    return True
                return False
            except Exception:
                return False
    
    def list_files(self, user_id: Optional[str] = None, prefix: str = "") -> List[str]:
        """
        List files in storage.
        
        Args:
            user_id: Optional user ID to filter files
            prefix: Optional prefix to filter files
            
        Returns:
            List of file paths
        """
        if self.storage_type == "s3":
            try:
                prefix_path = f"{user_id}/{prefix}" if user_id else prefix
                response = self.s3_client.list_objects_v2(
                    Bucket=self.bucket_name,
                    Prefix=prefix_path
                )
                
                files = []
                if 'Contents' in response:
                    for obj in response['Contents']:
                        files.append(f"s3://{self.bucket_name}/{obj['Key']}")
                return files
            except ClientError:
                return []
        else:  # local storage
            try:
                search_path = os.path.join(self.local_storage_path, user_id or "", prefix)
                files = []
                if os.path.exists(search_path):
                    for root, dirs, filenames in os.walk(search_path):
                        for filename in filenames:
                            files.append(os.path.join(root, filename))
                return files
            except Exception:
                return []
    
    def file_exists(self, file_path: str) -> bool:
        """
        Check if a file exists in storage.
        
        Args:
            file_path: Path/URL of the file
            
        Returns:
            True if file exists, False otherwise
        """
        if self.storage_type == "s3":
            try:
                if file_path.startswith("s3://"):
                    parts = file_path.replace("s3://", "").split("/", 1)
                    bucket = parts[0]
                    key = parts[1] if len(parts) > 1 else ""
                else:
                    bucket = self.bucket_name
                    key = file_path
                
                self.s3_client.head_object(Bucket=bucket, Key=key)
                return True
            except ClientError:
                return False
        else:  # local storage
            return os.path.exists(file_path)

