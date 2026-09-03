"""Flask entry point for the whole backend.

Only the recommendation module (Module 03) ships in this delivery; the profiling and
prediction blueprints register the same way when their folders are dropped in.
"""
from flask import Flask, jsonify

try:
    from flask_cors import CORS
except ImportError:  # pragma: no cover
    CORS = None

from config import Config
from recommendation_module.routes import recommendation_bp


def create_app(config_class=Config):
    app = Flask(__name__)
    app.config.from_object(config_class)
    if CORS is not None:
        CORS(app, resources={r"/api/*": {"origins": "*"}})

    app.register_blueprint(recommendation_bp)
    # app.register_blueprint(profiling_bp)      # Module 01
    # app.register_blueprint(prediction_bp)     # Module 02

    @app.get("/")
    def root():
        return jsonify({
            "service": "AI-Based Student Performance Prediction System",
            "modules": ["recommendation"],
            "endpoints": [str(r) for r in app.url_map.iter_rules() if str(r).startswith("/api")],
        })

    return app


if __name__ == "__main__":
    create_app().run(host="0.0.0.0", port=5000, debug=True)
