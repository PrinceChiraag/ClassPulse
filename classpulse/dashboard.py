from flask import jsonify


def dashboard_summary(history):
    if not history:
        return {"average_attention": 0.0, "fatigue_alerts": 0, "active_students": 0}

    average_attention = sum(item["attention"] for item in history) / len(history)
    fatigue_alerts = sum(1 for item in history if item["fatigue"] > 0.75)
    return {
        "average_attention": round(average_attention, 1),
        "fatigue_alerts": fatigue_alerts,
        "active_students": len(history),
    }
