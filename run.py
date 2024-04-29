from app.app import create_app, socketio

app = create_app()

if __name__ == '__main__':
    socketio.run(app, port=8080, debug=True)
