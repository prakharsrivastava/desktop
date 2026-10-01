"""
e360_cloud_connect_python_client.protegrity_file_envelope_protection_service

Reconstructed from screen recording 2026-06-11_12-53-02.mp4 (fully legible).
Confidence: HIGH for the two methods below.

This is the MOCK implementation shown in the video. In the real deployment this
class wraps Protegrity's file-envelope encryption; here it just prefixes the
bytes so the rest of the pipeline can be exercised without real keys.
"""

from io import BytesIO


class ProtegrityFileEnvelopeProtectionService:
    def __init__(self, logger=None):
        # `logger` was referenced by the methods in the recording; constructor
        # body was not shown, so this is the minimal faithful signature.
        self.logger = logger

    async def file_envelope_encryption_stream(self, data_stream: BytesIO) -> BytesIO:
        data = data_stream.read()

        # In a real implementation, this would encrypt the data.
        # For now, we'll just return it as-is with a prefix to indicate it's mock.
        mock_encrypted = b"MOCK_ENCRYPTED:" + data

        return BytesIO(mock_encrypted)

    async def file_envelope_decryption_stream(self, encrypted_stream: BytesIO) -> BytesIO:
        """Mock decryption - removes mock prefix"""
        self.logger.warning("MOCK DECRYPTION: Data was not actually encrypted!")

        # Read the encrypted data
        encrypted_stream.seek(0)
        encrypted_data = encrypted_stream.read()

        # Remove the mock prefix if present
        if encrypted_data.startswith(b"MOCK_ENCRYPTED:"):
            decrypted_data = encrypted_data[len(b"MOCK_ENCRYPTED:"):]
        else:
            # If no prefix, assume it's already decrypted
            decrypted_data = encrypted_data

        return BytesIO(decrypted_data)
