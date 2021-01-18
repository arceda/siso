package com.example.drowsinessdetector;

import android.content.Context;
import android.media.AudioManager;
import android.media.MediaPlayer;
import android.media.MediaRecorder;
import android.os.AsyncTask;
import android.util.Log;
import android.hardware.Camera;
import android.widget.FrameLayout;
import android.widget.Toast;

import org.json.JSONObject;

import java.io.BufferedOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.net.HttpURLConnection;
import java.net.URL;
import java.text.SimpleDateFormat;
import java.util.Date;

public class VideoSender extends AsyncTask<String, Void, Void> {

    File vid;
    URL serverUrl;
    CameraActivity mCameraActivity;
    Camera mCamera;
    MediaRecorder mrec; // for stopping the recorder before sending
    FrameLayout mFrameLayout;
    CameraPreview mPreview;

    String outfile_path; // where the video is saved on the Android phone
    String cacheDir; // temporary storage for video file
    int interval = 1000; // time between sending video ; in miliseconds ; also needs to be large enough so
    // the media recorder has some content (mrec.stop() fails if not enough content apparently...)

    HttpURLConnection urlConnection;

    boolean mBrokeStreak = false;

    // constructor ; store video to send to AWS
    public VideoSender(CameraActivity camActivity, CameraPreview preview, Camera cam,
                       String cacheDir, FrameLayout frameLayout) {
        this.mCameraActivity = camActivity;
        this.mPreview = preview;
        this.mCamera = cam;
        this.cacheDir = cacheDir;
        this.mFrameLayout = frameLayout;

        startMediaRecorder();
    }

    // when task is cancelled (triggered by pressing Stop button)
    @Override
    protected void onCancelled() {
        Log.d("VideoSender", "onCancelled");

        super.onCancelled();
        Log.d("VideoSender", "Terminating VideoSender and closing HTTP connection");
        try {
            urlConnection.disconnect();
            mrec.stop(); // stop recorder
            mrec.reset(); // reset configuration settings
            mrec.release(); // release media recorder

        } catch(Exception e) {
            Log.d("VideoSender", "Failed to disconnect HTTP connection and " +
                    "release media recorder");
            e.printStackTrace();
        }

        try {
            // lock camera
            mCamera.lock();
        } catch(Exception e) {
            Log.d("VideoSender", "Failed to lock camera");
        }

        try {
            mCamera.stopPreview();
            mCamera.release();
        } catch(Exception e) {
            Log.d("VideoSender", "Failed to release camera");
        }

    }

    protected void startMediaRecorder() {

        try{
            mrec = new MediaRecorder();
            // turn off recording sounds
            ((AudioManager)mCameraActivity.getSystemService(Context.AUDIO_SERVICE)).setStreamMute(AudioManager.STREAM_SYSTEM,true);
        } catch(Exception e) {
            Log.d("VideoSender", "Failed to create a new MediaRecorder");
            e.printStackTrace();
            return;
        }

        mCamera.unlock();

        // configure MediaRecorder
        mrec.setCamera(mCamera);
        mrec.setVideoSource(MediaRecorder.VideoSource.CAMERA);
        mrec.setOutputFormat(MediaRecorder.OutputFormat.MPEG_4);
        mrec.setVideoEncoder(MediaRecorder.VideoEncoder.DEFAULT);

        String timeStamp = new SimpleDateFormat("yyyyMMdd_HHmmss").format(new Date());
        this.outfile_path = this.cacheDir + "/" + "VID_" + timeStamp + ".mpeg";
        mrec.setOutputFile(this.outfile_path);
        Log.d("startCamera", "file_path: " + this.outfile_path);

        try {
            mrec.prepare();
            mrec.start();
            //Thread.sleep(5000);
        } catch(Exception e) {
            Log.d("VideoSender", "Failed to start media recorder.");
            e.printStackTrace();
        }

    }


    // parse JSON object received from HTTP connection
    protected boolean isDrowsy(HttpURLConnection urlConnection) {

        // Get JSON object from AWS response
        boolean isDrowsy = false;

        try {
            InputStream in = urlConnection.getInputStream();
            InputStreamReader inputStreamReader = new InputStreamReader(in);

            String jsonString = "";
            int inputStreamData = inputStreamReader.read();
            while (inputStreamData != -1) {
                char current = (char) inputStreamData;
                inputStreamData = inputStreamReader.read();
                jsonString += current;
            }
            in.close();

            Log.d("VideoSender", "Retrieved: " + jsonString);

            // Parse JSON object
            JSONObject jsonObject = new JSONObject(jsonString);
            isDrowsy = jsonObject.getBoolean("drowsy");

        } catch(Exception e) {
            Log.d("VideoSender", "Failed to create InputStream");
            e.printStackTrace();
            return false;
        }

        return isDrowsy;
    }

    // sends a video periodically
    // https://stackoverflow.com/questions/5017093/upload-video-from-android-to-server
    @Override
    protected Void doInBackground(String... params) {
        Log.d("VideoSender", "doInBackground()");

        int num_sent = 0; // only for debugging purposes
        OutputStream out;

        while(!isCancelled()) { // repeat until Stop button is pressed

            // record video for Interval time
            try {
                Log.d("VideoSender", "Sleeping for " + interval + " ms");
                Thread.sleep(interval);
            } catch(Exception e) {
                Log.d("VideoSender", "Failed to sleep...??");
                e.printStackTrace();
                break;
            }

            try {
                this.serverUrl = new URL(params[0]);
                // create a connection with server
                Log.d("VideoSender", "Server URL: " + serverUrl.toString());
                urlConnection = (HttpURLConnection) serverUrl.openConnection();
                urlConnection.setDoOutput(true);
                urlConnection.setRequestMethod("POST");
                urlConnection.addRequestProperty("Content-Type", "application/octet-stream");
                //urlConnection.setRequestProperty("connection", "Keep-Alive");
            } catch(Exception e) {
                Log.d("VideoSender", "Failed to create HTTP connection.");
                e.printStackTrace();
                return null;
            }

            try {
                Log.d("VideoSender", "Sending video #" + num_sent);

                // stop recorder
                try {
                    mrec.stop();
                } catch(Exception e) {
                    Log.d("VideoSender", "Stopped failed...?");
                    e.printStackTrace();
                }

                mrec.release();
                mCamera.lock();

                File vid = new File(this.outfile_path);
                if (!vid.isFile()) {
                    Log.e("VideoSender", "Video does not exist");
                    break;
                }

                // start the recorder again
                startMediaRecorder();

                Log.d("VideoSender", "length of video=" + vid.length());
                byte[] b = new byte[(int) vid.length()];
                //add the content length of the post data
                urlConnection.addRequestProperty("Content-Length", Integer.toString((int) vid.length()));

                FileInputStream fileInputStream = new FileInputStream(vid);
                fileInputStream.read(b);
                try {
                    out = urlConnection.getOutputStream();
                } catch(Exception e) {
                    Log.d("VideoSender", "Failed to create OutputStream");
                    break;
                }

                out = new BufferedOutputStream(out);
                out.write(b, 0, (int)vid.length()); // send to AWS
                out.flush();
                out.close();

                // Response from the server (code and message)
                int serverResponseCode = urlConnection.getResponseCode();
                if(serverResponseCode != 200) {
                    Log.d("VideoSender", "Server failed.");
                    break;
                }

                num_sent++; // number of videos sent

                boolean isDrowsy = this.isDrowsy( urlConnection );
                Log.d("VideoSender", "isDrowsy: " + isDrowsy);
                if(isDrowsy) {
                    // create notification message
                    mCameraActivity.runOnUiThread(new Runnable() {
                        @Override
                        public void run() {
                            Log.d("VideoSender", "Creating a toast message.");
                            CharSequence text = "Drowsiness detected. Please pull over for your safety.";
                            int duration = Toast.LENGTH_SHORT;
                            Toast toast = Toast.makeText(mCameraActivity, text, duration);
                            toast.show();

                            // make sound effect to alert the driver
                            MediaPlayer mp = MediaPlayer.create(mCameraActivity.getApplicationContext(), R.raw.alert);
                            mp.start();
                            try {
                                Thread.sleep(interval);
                            } catch(Exception e) {
                                Log.d("VideoSender", "Failed to sleep...");
                            }
                            mp.pause();
                            mp.stop();
                            mp.release();
                        }
                    }); // notification message

                    if(!mBrokeStreak) {
                        mBrokeStreak = true;
                        mCameraActivity.saveResult(mBrokeStreak);
                    }
                }

                urlConnection.disconnect();
            } catch (Exception e) {
                Log.d("VideoSender", "Failed to send video");
                e.printStackTrace();
                break;
            }
        } // while(1) ; end

        return null;
    }

    // This old version only sends one video at a time
//    @Override
//    protected Void doInBackground(String... params) {
//        Log.d("VideoSender", "doInBackground()");
//        try{
//            URL url = new URL(params[0]);
//            Log.d("VideoSender", "URL received: " + url.toString());
//
//            HttpURLConnection urlConnection  = (HttpURLConnection) url.openConnection();
//            urlConnection.setDoOutput(true);
//            //urlConnection.setChunkedStreamingMode(0);
//            //urlConnection.setFixedLengthStreamingMode(vid.length());
//            urlConnection.setRequestMethod("POST");
//            urlConnection.addRequestProperty("Content-Length", Integer.toString((int) vid.length()));
//            //urlConnection.addRequestProperty("Content-Type", "video/mpeg");
//            urlConnection.addRequestProperty("Content-Type", "application/octet-stream");
//
//            Log.d("VideoSender", "length of video=" + this.vid.length());
//            byte[] b = new byte[(int) vid.length()];
//            FileInputStream fileInputStream = new FileInputStream(vid);
//            int read = fileInputStream.read(b);
//            Log.d("VideoSender", "Read " + read + " bytes");
//            OutputStream out = urlConnection.getOutputStream();
//            out = new BufferedOutputStream(out);
//            out.write(b, 0, (int)vid.length());
//            out.flush();
//            out.close();
//
//            // Response from the server (code and message)
//            int serverResponseCode = urlConnection.getResponseCode();
//            String serverResponseMessage = urlConnection.getResponseMessage();
//            Log.d("VideoSender", "Server response: " + serverResponseMessage +
//                    "; code=" + Integer.toString(serverResponseCode));
//
//            // Get JSON object from AWS response
//            InputStream in = urlConnection.getInputStream();
//            InputStreamReader inputStreamReader = new InputStreamReader(in);
//
//            String jsonString = "";
//            int inputStreamData = inputStreamReader.read();
//            while (inputStreamData != -1) {
//                char current = (char) inputStreamData;
//                inputStreamData = inputStreamReader.read();
//                jsonString += current;
//            }
//
//            Log.d("VideoSender", "Retrieved: " + jsonString);
//
//            // Parse JSON object
//            JSONObject jsonObject = new JSONObject(jsonString);
//            boolean isDrowsy = jsonObject.getBoolean("drowsy");
//            Log.d("VideoSender", "isDrowsy: " + isDrowsy);
//
//            urlConnection.disconnect();
//        } catch(Exception e) {
//            Log.d("VideoSender", "Failed to send video!");
//            Log.d("VideoSender", "Exception: " + e);
//            e.printStackTrace();
//            System.out.println(e);
//        }
//
//        return null;
//    }

}
