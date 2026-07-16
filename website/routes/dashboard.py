from flask import Blueprint, render_template

dashboard_bp = Blueprint('dashboard', __name__)

@dashboard_bp.route('/', methods=['GET'])
def dashboard():
    stats={
        'document': 0,
        'entities': 0,
        'relationship': 0
    }
    return render_template('dashboard.html', stats=stats)