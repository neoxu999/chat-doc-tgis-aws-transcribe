import asyncio
import os

import aiofile
import tempfile

from amazon_transcribe.auth import StaticCredentialResolver
from amazon_transcribe.client import TranscribeStreamingClient
from amazon_transcribe.handlers import TranscriptResultStreamHandler
from amazon_transcribe.model import TranscriptEvent
from amazon_transcribe.utils import apply_realtime_delay

"""
Here's an example of a custom event handler you can extend to
process the returned transcription results as needed. This
handler will simply print the text out to your interpreter.
"""

SAMPLE_RATE = 32000
BYTES_PER_SAMPLE = 2
CHANNEL_NUMS = 1

# An example file can be found at tests/integration/assets/test.wav
AUDIO_PATH = "/home/neoxu/PycharmProjects/chat-doc/test/preamble10.wav"
CHUNK_SIZE = 1024 * 8
REGION = "ap-southeast-2"
AWS_ACCESS_KEY_ID = os.environ.get("AWS_ACCESS_KEY_ID")
AWS_ACCESS_KEY = os.environ.get("AWS_ACCESS_KEY")


class MyEventHandler(TranscriptResultStreamHandler):

    def __init__(self, transcript_result_stream):
        super().__init__(transcript_result_stream)
        self.full_transcript = []  # Initialize a list to hold parts of the tr

    async def handle_transcript_event(self, transcript_event: TranscriptEvent):
        # This handler can be implemented to handle transcriptions as needed.
        # Here's an example to get started.
        results = transcript_event.transcript.results
        for result in results:
            if result.is_partial:
                continue  # Skip partial results
            for alt in result.alternatives:
                self.full_transcript.append(alt.transcript)

    async def handle_events(self):
        await super().handle_events()
        # Once all events are handled (i.e., the transcription is complete),
        # print the full transcript as a single string.
        full_transcript = ' '.join(self.full_transcript).strip()
        print("Finished:" + full_transcript)
        return full_transcript


async def basic_transcribe(audio_path):
    # Setup up our client with our chosen AWS region
    static_credential_resolver = StaticCredentialResolver(
        access_key_id=AWS_ACCESS_KEY_ID,
        secret_access_key=AWS_ACCESS_KEY,
    )

    client = TranscribeStreamingClient(region=REGION, credential_resolver=static_credential_resolver)

    # Start transcription to generate our async stream
    stream = await client.start_stream_transcription(
        language_code="en-US",
        media_sample_rate_hz=SAMPLE_RATE,
        media_encoding="pcm",
    )

    async def write_chunks(audio_path):
        # NOTE: For pre-recorded files longer than 5 minutes, the sent audio
        # chunks should be rate limited to match the realtime bitrate of the
        # audio stream to avoid signing issues.
        async with aiofile.AIOFile(audio_path, "rb") as afp:
            reader = aiofile.Reader(afp, chunk_size=CHUNK_SIZE)
            await apply_realtime_delay(
                stream, reader, BYTES_PER_SAMPLE, SAMPLE_RATE, CHANNEL_NUMS
            )
        await stream.input_stream.end_stream()

    # Instantiate our handler and start processing events
    handler = MyEventHandler(stream.output_stream)
    full_transcript = await asyncio.gather(write_chunks(audio_path), handler.handle_events())
    return full_transcript[1]  # Return the full transcript


# loop = asyncio.get_event_loop()
# transcript = loop.run_until_complete(basic_transcribe(AUDIO_PATH))
# print("Get the transcript:" + transcript)
# loop.close()
