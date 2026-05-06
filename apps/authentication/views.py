"""
Authentication Views
"""

from django.shortcuts import render, redirect
from django.contrib.auth import login, logout, authenticate
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.views.decorators.http import require_http_methods
from .models import User, UserPreferences, UserBehaviorProfile


@require_http_methods(["GET", "POST"])
def register_view(request):
    """User registration"""
    if request.method == 'POST':
        username = request.POST.get('username')
        email = request.POST.get('email')
        password = request.POST.get('password')
        password_confirm = request.POST.get('password_confirm')
        
        # Validation
        if password != password_confirm:
            messages.error(request, 'Passwords do not match')
            return render(request, 'authentication/register.html')
        
        if User.objects.filter(username=username).exists():
            messages.error(request, 'Username already exists')
            return render(request, 'authentication/register.html')
        
        if User.objects.filter(email=email).exists():
            messages.error(request, 'Email already registered')
            return render(request, 'authentication/register.html')
        
        # Create user
        user = User.objects.create_user(
            username=username,
            email=email,
            password=password
        )
        
        # Create user preferences
        UserPreferences.objects.create(user=user)
        
        # Create behavior profile
        UserBehaviorProfile.objects.create(user=user)
        
        messages.success(request, 'Registration successful! Please login.')
        return redirect('login')
    
    return render(request, 'authentication/register.html')


@require_http_methods(["GET", "POST"])
def login_view(request):
    """User login"""
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')
        
        user = authenticate(request, username=username, password=password)
        
        if user is not None:
            login(request, user)
            messages.success(request, f'Welcome back, {user.username}!')
            return redirect('dashboard')
        else:
            messages.error(request, 'Invalid username or password')
    
    return render(request, 'authentication/login.html')


@login_required
def logout_view(request):
    """User logout"""
    logout(request)
    messages.success(request, 'You have been logged out successfully')
    return redirect('home')


@login_required
def profile_view(request):
    """User profile"""
    user = request.user
    preferences = user.preferences
    behavior_profile = user.behavior_profile
    
    context = {
        'user': user,
        'preferences': preferences,
        'behavior_profile': behavior_profile
    }
    
    return render(request, 'authentication/profile.html', context)


@login_required
@require_http_methods(["POST"])
def update_preferences(request):
    """Update user preferences"""
    preferences = request.user.preferences
    
    preferences.preferred_language = request.POST.get('preferred_language', preferences.preferred_language)
    preferences.theme = request.POST.get('theme', preferences.theme)
    preferences.enable_voice_input = request.POST.get('enable_voice_input') == 'on'
    preferences.enable_camera_emotion = request.POST.get('enable_camera_emotion') == 'on'
    preferences.enable_notifications = request.POST.get('enable_notifications') == 'on'
    preferences.save()
    
    messages.success(request, 'Preferences updated successfully')
    return redirect('profile')
