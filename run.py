from app.app import create_app, socketio

app = create_app()

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0' ,port=8080)
    
    # For debugging purpose. The websocket will not woek well
    # socketio.run(app, host='0.0.0.0' ,port=8080, debug=True)
    
# uWSGI running:
# uwsgi --http :8080 --gevent 1000 --enable-threads --lazy-apps --http-websockets --master --wsgi-file run.py --callable app
