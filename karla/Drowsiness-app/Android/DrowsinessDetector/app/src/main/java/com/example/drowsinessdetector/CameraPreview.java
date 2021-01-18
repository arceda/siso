package com.example.drowsinessdetector;

import android.content.Context;
import android.hardware.Camera;
import android.util.Log;
import android.view.SurfaceView;
import android.view.SurfaceHolder;

import java.io.IOException;

// this class allows user to see what the camera is capturing
// SurfaceView captures live video data from the camera
// SurfaceHolder.Callback captures the callback events for creating/destroying the View
public class CameraPreview extends SurfaceView implements SurfaceHolder.Callback {
    private SurfaceHolder mHolder; // SurfaceView != SurfaceHolder
    private Camera mCamera;

    // constructor
    public CameraPreview(Context context, Camera camera) {
        super(context);
        mCamera = camera;

        mHolder = getHolder(); // gets reference to the View that is holding this CameraView
        mHolder.addCallback(this);
        mHolder.setType(SurfaceHolder.SURFACE_TYPE_PUSH_BUFFERS); // legacy check
    }

    // Tell camera where to draw the preview
    public void surfaceCreated(SurfaceHolder holder) {
        try {
            mCamera.setPreviewDisplay(holder);
            mCamera.startPreview();
        } catch(IOException e) {
            Log.d("CameraPreview", "surfaceCreated(): Error in setting camera preview: " + e.getMessage());
        }
    }

    public void surfaceDestroyed(SurfaceHolder holder) {
        // empty
    }

    // When the preview changes (ex. rotates)
    public void surfaceChanged(SurfaceHolder holder, int format, int w, int h) {
        if(mHolder.getSurface() == null) return;

        // stop preview before making changes
        try {
            mCamera.stopPreview();
        } catch(Exception e) {
            Log.d("CameraPreview", "surfaceChanged(): Tried to stop a non-existent preview.");
        }

        // set preview size and make any resize, rotate or reformatting
        try {
            mCamera.setPreviewDisplay(mHolder);
            mCamera.startPreview();
        } catch(Exception e) {
            Log.d("CameraPreview" , "surfaceChanged(): Error starting camera preview: " + e.getMessage());
        }
    }

    public SurfaceHolder getSurfaceHolder() {
        return mHolder;
    }

}
