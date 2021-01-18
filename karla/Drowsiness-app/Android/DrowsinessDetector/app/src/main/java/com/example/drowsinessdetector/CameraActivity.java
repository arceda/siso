package com.example.drowsinessdetector;

import androidx.appcompat.app.AppCompatActivity;

import android.content.Context;
import android.content.Intent;
import android.hardware.Camera;
import android.os.AsyncTask;
import android.os.Bundle;
import android.os.PowerManager;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.FrameLayout;

public class CameraActivity extends AppCompatActivity {

    // to send result to HomeActivity
    public static final String EXTRA_REPLY =
            "com.example.android.twoactivities.extra.REPLY";

    public static final int FRONT_CAMERA = 1;

    private Camera mCamera;
    private CameraPreview mPreview;

    private boolean mBrokeStreak;

    PowerManager.WakeLock mWakelock; // prevent screen from sleeping

    // AWS url
    private String serverUrl = "http://ec2-3-16-1-156.us-east-2.compute.amazonaws.com:8000";
    // private String serverUrl = "http://192.168.1.85:8000";

    AsyncTask<String, Void, Void> sender;

    private boolean isRecording = false;

    // get Camera instance
    public static Camera getCamera() {
        Camera c = null;
        try {
            c = Camera.open(FRONT_CAMERA);
        } catch(Exception e) {
            Log.d("getCamera", "Camera does not exist.");
        }

        return c; // return Camera instance
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_camera);
        Log.d("CameraActiivty", "created CameraActivity");

        mCamera = getCamera();
        if(mCamera == null) {
            Log.d("CameraActivity", "Failed to get camera onCreate()");
            return;
        }

        // Create our Preview view and set it as the content of our activity.
        mPreview = new CameraPreview(this, mCamera);
        FrameLayout preview = findViewById(R.id.camera_preview);
        preview.addView(mPreview);

        // prevent screen from sleeping
        PowerManager pm = (PowerManager) getSystemService(Context.POWER_SERVICE);
        mWakelock = pm.newWakeLock(PowerManager.SCREEN_DIM_WAKE_LOCK, "CameraActivity:wakeLock");
        mWakelock.acquire();
    }

    public void returnToMain(View view) {
        Log.d("CameraActivity", "Returning to Main.");
        if(mCamera != null) {
            mCamera.stopPreview();
            mCamera.release();
        }

        mWakelock.release();

        // send result of drowsiness recording back to HomeActivity
        Intent result = new Intent();
        Log.d("CameraActivity", "Sending " + mBrokeStreak + " to HomeActivity");
        result.putExtra(EXTRA_REPLY, Boolean.toString(mBrokeStreak));
        setResult(RESULT_OK, result);

        finish(); // return to HomeActivity
    }

    private boolean startCamera() {

        if(mCamera == null) {
            Log.d("startCamera", "Failed to get camera.");
            return false;
        }

        Log.d("startCamera", "Camera is ready to record.");

        // Start the background process VideoSender, that periodically sends video to server
        // VideoSender will create MediaRecorder objects
        try {
            this.sender = new VideoSender(this, mPreview, mCamera,
                    this.getCacheDir().toString(), (FrameLayout) findViewById(R.id.camera_preview));
            sender.execute(this.serverUrl);
        } catch (Exception e) {
            Log.d("startCamera", "Failed to send video");
            return false;
        }

        return true; // success
    }

    private void setCameraButtonText(Button b, String s) {
        b.setText(s);
    }

    // Use one button for start, stop, record camera
    public void toggleCamera(View view) {
        Log.d("toggleCamera", "toggleCamera() was called.");
        // Create a listener for camera button
        final Button cameraButton = (Button) findViewById(R.id.camera_button);
        cameraButton.setOnClickListener(
                new View.OnClickListener() {
                    @Override
                    public void onClick(View v) {
                        if(isRecording) {
                            // terminate background process VideoSender
                            sender.cancel(true);
                            setCameraButtonText(cameraButton, "Record");
                            isRecording = false;
                            // return to home screen
                            Log.d("toggleCamera", "Stopped recording");
                            returnToMain(findViewById(android.R.id.content));
                        } else {
                            if(startCamera()) {
                                setCameraButtonText(cameraButton, "Stop");
                                isRecording = true;
                                Log.d("toggleCamera", "Started recording");
                            } else { // failed
                                // stopCamera();
                                Log.d("toggleCamera", "Failed");
                            } // failed ; end
                        }
                    } // onClick() ; end
                } // new View.OnClickListener() ; end
        ); // cameraButton.setOnClickListener ; end
    }

    public void saveResult(boolean brokeStreak) {
        mBrokeStreak = brokeStreak;
        Log.d("CameraActivity", "set mBrokeStreak: " + mBrokeStreak);
    }

}