package com.example.drowsinessdetector;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import android.Manifest;
import android.content.Context;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.TextView;

import com.google.firebase.auth.FirebaseAuth;
import com.google.firebase.database.DataSnapshot;
import com.google.firebase.database.DatabaseError;
import com.google.firebase.database.DatabaseReference;
import com.google.firebase.database.FirebaseDatabase;
import com.google.firebase.database.ValueEventListener;

public class HomeActivity extends AppCompatActivity {

    public static final int TEXT_REQUEST = 1;
    private final int CAMERA_PERMISSION_REQUEST = 0;

    private Button buttonLogout;

    FirebaseAuth firebaseAuth;
    FirebaseDatabase mFirebaseDatabase;
    DatabaseReference mDbReference;
    String uname;

    // Check if this device has a camera
    private boolean deviceHasCamera(Context context) {
        // no camera on this device
        return context.getPackageManager().hasSystemFeature(PackageManager.FEATURE_CAMERA); // this device has a camera
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {

        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_home);

        mDbReference = mFirebaseDatabase.getInstance().getReference("streak");

        Intent intent = getIntent(); // get Activity that initiated this Activity


        Log.d("HomeActivity", "onCreate(): Has camera: " + deviceHasCamera(this));

        // Logout
        buttonLogout.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                // Signout from firebase
                firebaseAuth.getInstance().signOut();
                // After signout, go to the Login page
                Intent intent = new Intent(HomeActivity.this, MainActivity.class);
                startActivity(intent);
            }
        });

    }


    // Called when user decides to grant a permission or not
    @Override
    public void onRequestPermissionsResult(int requestCode,
                                           String[] permissions, int[] grantResults) {
        switch(requestCode) {
            case CAMERA_PERMISSION_REQUEST: {
                // If request is cancelled, the result arrays are empty.
                if(grantResults.length > 0
                        && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                    Log.d("HomeActivity", "Permission was granted. Yay!");
                } else {
                    Log.d("HomeActivity", "Permission was denied. Boohoo.");
                }
                return;
            }
        }
        // Check other permissions here
    }

    // Check if the app has permission to use the camera
    // The user can always disable permission anytime ; can't assume we always have permission
    private boolean getCameraPermission() {

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
                != PackageManager.PERMISSION_GRANTED) {
            // Permission is not granted
            Log.d("HomeActivity", "Permission to use the camera was not granted.");

            // Ask user for permission to use the camera (pop-up message)
            ActivityCompat.requestPermissions(this,
                    new String[]{Manifest.permission.CAMERA}, CAMERA_PERMISSION_REQUEST);
            Log.d("HomeActivity", "CAMERA_PERMISSION_REQUEST: " + CAMERA_PERMISSION_REQUEST);

            // Check result of pop-up message
            return ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) != PackageManager.PERMISSION_DENIED;
        }

        return true;
    }

    public void startCameraActivity(View view) {
        Log.d("HomeActivity", "startCamera()");

        // check if this device has a camera
        if(!deviceHasCamera(this)) {
            Log.d("HomeActivity", "No camera on this device.");
            // return to HomeActivity
            finish();
        }

        if(!getCameraPermission()) {
            Log.d("HomeActivity", "Failed to get permission from user.");
            // return to HomeActivity
            finish();
        }

        // start CameraActivity
        Log.d("HomeActivity", "Ready to start CameraActivity.");
        Intent intent = new Intent(this, CameraActivity.class);
        // startActivity(intent);
        startActivityForResult(intent, TEXT_REQUEST);
    }


    // retrieves result of video recording (drowsy or not)
    @Override
    public void onActivityResult(int requestCode, int resultCode, Intent data) {
        mDbReference = mFirebaseDatabase.getInstance().getReference("streak");
        super.onActivityResult(requestCode, resultCode, data);
        if (requestCode == TEXT_REQUEST) {
            if (resultCode == RESULT_OK) {
                final String brokeStreak = data.getStringExtra(CameraActivity.EXTRA_REPLY);
                Log.d("HomeActivity", "brokeStreak: " + brokeStreak);

                mDbReference.orderByChild("user").equalTo(uname).addListenerForSingleValueEvent(new ValueEventListener() {
                    @Override
                    public void onDataChange(@NonNull DataSnapshot dataSnapshot) {

                        for(DataSnapshot snapshot : dataSnapshot.getChildren())
                        {
                            if(brokeStreak.equalsIgnoreCase("true"))
                            {
                                //streak value set to 0 in the database when brokenStreak value is true
                                snapshot.getRef().child("streak").setValue(0);
                            }
                            else
                            {
                                Streak records = snapshot.getValue(Streak.class);
                                int streak_value = records.getStreak();
                                //streak value is incremented by 1 when brokenStreak value is false
                                snapshot.getRef().child("streak").setValue(streak_value+1);
                            }
                            Streak records = snapshot.getValue(Streak.class);
                            int streak_value = records.getStreak();

                            Log.d("HomeActivity", "Streak count in the database is " + streak_value);
                        }
                    }

                    @Override
                    public void onCancelled(@NonNull DatabaseError databaseError) {

                    }
                });



            }
        }
    }
}