
## Disfluency Detection Methodology

### Detection Approach
1. **Pattern Matching**: Used exact string matching and regex for 196 predefined patterns
2. **Text Normalization**: Unicode normalization and whitespace cleaning
3. **Confidence Scoring**: Based on exact match, case sensitivity, and character similarity
4. **Context Extraction**: 50-character context window around detections

### Audio Clipping
1. **Timestamp Estimation**: Uniform speech rate assumption (characters/duration)
2. **Segment Extraction**: Using librosa with 0.5s padding around disfluencies
3. **File Generation**: 16kHz WAV files with descriptive naming

### Preprocessing Steps
1. **Audio Processing**: Resampling to 16kHz, normalization, silence trimming
2. **Text Cleaning**: Unicode NFC normalization, extra whitespace removal
3. **Quality Filtering**: Minimum confidence threshold, context validation

### Limitations & Future Improvements
- **Timestamp Accuracy**: Could be improved with forced alignment
- **Pattern Coverage**: Additional patterns could be learned from data
- **Context Sensitivity**: Could consider linguistic context for better detection
