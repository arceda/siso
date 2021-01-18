package com.example.drowsinessdetector;

public class Streak {

    public String user;
    public int streak;

    public Streak(){};

    public Streak(String user, int streak)
    {
        this.user = user;
        this.streak = streak;
    }

    public void setUser(String user) {
        this.user = user;
    }

    public void setStreak(int streak) {
        this.streak = streak;
    }

    public String getUser() {
        return user;
    }

    public int getStreak() {
        return streak;
    }
}
