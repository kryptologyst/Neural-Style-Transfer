# Neural Style Transfer - Modern Implementation

A comprehensive neural style transfer application featuring multiple state-of-the-art algorithms, modern UI, and production-ready architecture.

## Features

- **Multiple Algorithms**: AdaIN (fast) and optimization-based (high quality) style transfer
- **Modern UI**: Beautiful Streamlit interface with real-time preview
- **REST API**: FastAPI backend for integration and batch processing
- **Template Gallery**: Pre-built artistic style templates
- **History Tracking**: SQLite database for transfer history and analytics
- **Batch Processing**: Process multiple images simultaneously
- **Performance Analytics**: Detailed statistics and processing metrics

## Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd 0090_Image_style_transfer

# Install dependencies
pip install -r requirements.txt

# Initialize database with sample templates
python src/database/models.py
```

### Running the Application

#### Option 1: Streamlit UI (Recommended)
```bash
streamlit run src/ui/streamlit_app.py
```

#### Option 2: FastAPI Backend
```bash
cd src/api
python main.py
# or
uvicorn main:app --host 0.0.0.0 --port 8000
```

## Architecture

```
src/
├── models/
│   └── neural_style_transfer.py    # Core ML models (AdaIN, Fast NST)
├── database/
│   └── models.py                   # SQLAlchemy models and mock data
├── api/
│   └── main.py                     # FastAPI REST endpoints
└── ui/
    └── streamlit_app.py           # Modern Streamlit interface
```

## Algorithms

### 1. AdaIN (Adaptive Instance Normalization)
- **Speed**: Very Fast (seconds)
- **Quality**: Good
- **Use Case**: Real-time applications, quick previews

### 2. Optimization-based Transfer
- **Speed**: Slower (minutes)
- **Quality**: Excellent
- **Use Case**: High-quality final results

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/transfer` | POST | Upload content & style images for transfer |
| `/transfer/template/{id}` | POST | Use predefined template for transfer |
| `/templates` | GET | Get available style templates |
| `/history` | GET | Get transfer history with pagination |
| `/stats` | GET | Get application statistics |

### Example API Usage

```python
import requests

# Upload and transfer
files = {
    'content_file': open('content.jpg', 'rb'),
    'style_file': open('style.jpg', 'rb')
}
data = {'method': 'adain', 'alpha': 0.8}

response = requests.post('http://localhost:8000/transfer', files=files, data=data)
result = response.json()
```

## Style Templates

The application includes pre-built templates for popular artistic styles:

- **Van Gogh - Starry Night**: Classic swirling brushstrokes
- **Picasso - Cubist**: Geometric shapes and abstract forms
- **Monet - Water Lilies**: Impressionist soft colors
- **Kandinsky - Abstract**: Bold colors and geometric patterns
- **Japanese Ukiyo-e**: Traditional woodblock print style

## 🔧 Configuration

### Model Parameters

```python
# AdaIN Parameters
alpha = 1.0          # Style strength (0.0 - 1.0)
image_size = 512     # Output image size

# Optimization Parameters
steps = 300          # Number of optimization steps
style_weight = 1e10  # Style loss weight
content_weight = 1e5 # Content loss weight
```

### Database Configuration

```python
# Default SQLite (development)
DATABASE_URL = "sqlite:///style_transfer.db"

# PostgreSQL (production)
DATABASE_URL = "postgresql://user:pass@localhost/styledb"
```

## Performance

| Method | Image Size | Processing Time | Quality |
|--------|------------|----------------|---------|
| AdaIN | 512x512 | ~2-5 seconds | Good |
| AdaIN | 1024x1024 | ~5-10 seconds | Good |
| Optimization | 512x512 | ~2-5 minutes | Excellent |
| Optimization | 1024x1024 | ~5-15 minutes | Excellent |

*Times measured on NVIDIA RTX 3080*

## 🛠️ Development

### Project Structure

```
├── src/                    # Source code
│   ├── models/            # ML models and algorithms
│   ├── database/          # Database models and operations
│   ├── api/              # REST API endpoints
│   └── ui/               # User interface
├── assets/               # Style templates and static files
├── uploads/              # Temporary upload directory
├── outputs/              # Generated results
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

### Adding New Style Templates

```python
from src.database.models import DatabaseManager

db = DatabaseManager()
db.add_style_template(
    name="New Style",
    description="Description of the artistic style",
    style_image_path="assets/styles/new_style.jpg",
    category="modern"
)
```

### Extending with New Algorithms

1. Implement your algorithm in `src/models/neural_style_transfer.py`
2. Add API endpoint in `src/api/main.py`
3. Update UI in `src/ui/streamlit_app.py`

## Technical Details

### Dependencies

- **PyTorch**: Deep learning framework
- **torchvision**: Pre-trained models and transforms
- **FastAPI**: Modern web framework for APIs
- **Streamlit**: Interactive web applications
- **SQLAlchemy**: Database ORM
- **Pillow**: Image processing
- **OpenCV**: Computer vision operations

### Model Architecture

The application uses a pre-trained VGG19 network for feature extraction:

- **Encoder**: VGG19 features up to `relu4_1`
- **Decoder**: Upsampling network with reflection padding
- **AdaIN**: Adaptive instance normalization for style transfer
- **Loss Functions**: Perceptual loss, style loss, content loss

## Deployment

### Docker Deployment

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY src/ ./src/
COPY assets/ ./assets/

EXPOSE 8000
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Cloud Deployment

The application is ready for deployment on:
- **Heroku**: Use `Procfile` with gunicorn
- **AWS**: Deploy with ECS or Lambda
- **Google Cloud**: Use Cloud Run or App Engine
- **Azure**: Deploy with Container Instances

## License

MIT License - see LICENSE file for details.

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## References

- [A Neural Algorithm of Artistic Style](https://arxiv.org/abs/1508.06576)
- [Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization](https://arxiv.org/abs/1703.06868)
- [Perceptual Losses for Real-Time Style Transfer and Super-Resolution](https://arxiv.org/abs/1603.08155)

## Support

For issues and questions:
1. Check the [Issues](../../issues) page
2. Create a new issue with detailed description
3. Include error logs and system information



# Neural-Style-Transfer
