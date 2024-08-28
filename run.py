from app.app import create_app

app = create_app() 

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8080)

    
    # For debugging purpose. The websocket will not woek well
    # uvicorn.run(app, host='0.0.0.0' ,port=8080, debug=True)
    
